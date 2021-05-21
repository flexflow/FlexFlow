/* Copyright 2017 Stanford, NVIDIA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::concat(int n,
                       const Tensor* tensors,
                       int axis,
                       const char *name)
{
  Concat *cat = new Concat(*this, n, tensors, axis, name);
  layers.push_back(cat);
  return cat->outputs[0];
}

Concat::Concat(FFModel& model,
               int _n, const Tensor* _tensors,
               int _axis,
               const char* name)
: Op(model, OP_CONCAT, name, _n, _tensors), axis(_axis)
{
  //TODO: swich to use the Legion dim ordering
  int num_dim = inputs[0].numDim;
  outputs[0].numDim = num_dim;
  for (int i = 0; i < num_dim; i++)
    outputs[0].adim[i] = inputs[0].adim[i];
  for (int i = 1; i < numInputs; i++)
    for (int j = 0; j < num_dim; j++) {
      if (j != num_dim - 1 - axis)
        assert(inputs[i].adim[j] == outputs[0].adim[j]);
      else
        outputs[0].adim[j] += inputs[i].adim[j];
    }
  numOutputs = 1;
  numWeights = 0;
}

void Concat::create_weights(FFModel& model)
{
  // DO nothing
}

void Concat::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = model.get_or_create_task_is(inputs[0].numDim, pcname);

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  int dims[MAX_TENSOR_DIM], num_dim = inputs[0].numDim;
  assert(num_dim == domain.get_dim());
  for (int i = 0; i < num_dim; i++)
    dims[i] = inputs[0].adim[num_dim-1-i];
  for (int i = 1; i < numInputs; i++)
    for (int j = 0; j < num_dim; j++) {
      if (j != axis)
        assert(inputs[i].adim[num_dim-1-j] == dims[j]);
      else
        dims[j] += inputs[i].adim[num_dim-1-j];
    }
  //for (int i = 0; i < num_dim; i++)
    //printf("concat: dim[%d] = %d\n", i, dims[i]);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> part_rect = domain; \
      outputs[0] = model.create_tensor<DIM>(dims, DT_FLOAT, this); \
      outputs[0].owner_op = this; \
      outputs[0].owner_idx = 0; \
      for (int i = 0; i < numInputs; i++) { \
        Rect<DIM> input_rect = runtime->get_index_partition_color_space( \
            ctx, inputs[i].part.get_index_partition()); \
        if (input_rect == part_rect) { \
          input_lps[i] = inputs[i].part; \
          input_grad_lps[i] = inputs[i].part_grad; \
        } else { \
          model.create_disjoint_partition<DIM>(inputs[i], \
              IndexSpaceT<DIM>(task_is), input_lps[i], input_grad_lps[i]); \
        } \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
    }
  }
}

void Concat::init_meta(ConcatMeta *m) const
{
  m->axis = this->outputs[0].numDim - 1 - this->axis;
}

__host__
OpMeta* Concat::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  Concat* cc = (Concat*) task->args;
  FFHandler handler = *((const FFHandler*) task->local_args);
  ConcatMeta* m = new ConcatMeta(handler);
  // Note that our internal axis index ordering is opposite to other frameworks
  cc->init_meta(m);
  m->profiling = cc->profiling;
  return m;
}

void Concat::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      ParallelConfig pc; \
      std::string pcname = name; \
      ff.config.find_parallel_config(DIM, pcname, pc); \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        FFHandler handle = ff.handlers[pc.device_ids[idx++]]; \
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(CONCAT_INIT_TASK_ID, task_is,
    TaskArgument(this, sizeof(Concat)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i + 1, FID_DATA);
  }
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i], 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, inputs[i].region_grad));
    launcher.add_field(i + numInputs + 1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        meta[idx++] = fm.get_result<OpMeta*>(*it); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int N>
void calc_blk_size(coord_t& num_blocks,
                   coord_t& blk_size,
                   Rect<N> rect,
                   int axis)
{
  num_blocks = 1;
  blk_size = 1;
  for (int d = 0; d < N; d++) {
    if (d <= axis)
      blk_size *= (rect.hi[d] - rect.lo[d] + 1);
    else
      num_blocks *= (rect.hi[d] - rect.lo[d] + 1);
  }
}

/*static*/
void Concat::forward_kernel(float* output,
                            float const * const *inputs,
                            int num_inputs,
                            int axis,
                            const Domain& out_domain,
                            const Domain* in_domain,
                            cudaStream_t stream)
{
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (out_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = out_domain; \
      calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis); \
      for (int i = 0; i < num_inputs; i++) { \
        rect = in_domain[i]; \
        coord_t input_num_blocks = 1; \
        calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis); \
        assert(input_num_blocks == num_blocks); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }

  for (int i = 0; i < num_inputs; i++) {
    copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS, 0, stream>>>(
        output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
    //printf("output = %x num_blocks=%d output_blk_size=%d input_blk_size[%d]=%d\n",
    //       output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    output += input_blk_sizes[i];
  }
}

/*
  regions[0](O): output
  regions[1..numInputs](I): inputs
*/
void Concat::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  // Note that our internal axis index ordering is opposite to other frameworks
  int axis = cc->outputs[0].numDim - 1 - cc->axis;
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(out_domain.get_dim() == cc->outputs[0].numDim);
  Domain in_domain[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    in_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i+1].region.get_index_space());
  float *output = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float *inputs[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    inputs[i] = helperGetTensorPointerRO<float>(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
  
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  cudaEvent_t t_start, t_end;
  if (cc->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  forward_kernel(output, inputs, cc->numInputs, axis, out_domain, in_domain, stream);
  if (cc->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<4, float>(output - output_blk_size, output_rect, "[Concat:forward:output]");
    //printf("output_blk_size=%zu\n", output_blk_size);
    //print_tensor<4, float>(inputs[0], input_rect[0], "[Concat:forward:input0]");
    //print_tensor<4, float>(inputs[1], input_rect[1], "[Concat:forward:input1]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", cc->name, elapsed);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
  }
}

void Concat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(CONCAT_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Concat)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Concat::backward_kernel(const float* output_grad,
                             float** input_grads,
                             int num_inputs,
                             int axis,
                             const Domain& out_grad_domain,
                             const Domain* in_grad_domain,
                             cudaStream_t stream)
{
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(num_inputs <= MAX_NUM_INPUTS);
  switch (out_grad_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = out_grad_domain; \
      calc_blk_size<DIM>(num_blocks, output_blk_size, rect, axis); \
      for (int i = 0; i < num_inputs; i++) { \
        rect = in_grad_domain[i]; \
        coord_t input_num_blocks = 1; \
        calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], rect, axis); \
        assert(input_num_blocks == num_blocks); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }

  for (int i = 0; i < num_inputs; i++) {
    add_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS, 0, stream>>>(
        input_grads[i], output_grad, num_blocks, input_blk_sizes[i], output_blk_size);
    output_grad += input_blk_sizes[i];
  }

  //Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size - 1));
  //Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1, batch_size - 1));
  //print_tensor<2, float>(output_grad - output_blk_size, output_rect, "[Concat:backward:output]");
  //print_tensor<2, float>(input_grads[0], input_rect, "[Concat:backward:input0]");
}

/*
  regions[0](I): output_grad
  regions[1..numInputs](I/O): input_grad
*/
void Concat::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  // Note that our internal axis index ordering is opposite to other frameworks
  int axis = cc->outputs[0].numDim - 1 - cc->axis;
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(out_grad_domain.get_dim() == cc->outputs[0].numDim);
  Domain in_grad_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    in_grad_domains[i] = runtime->get_index_space_domain(
        ctx, task->regions[i+1].region.get_index_space());
  const float *output_grad = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *input_grads[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    input_grads[i] = helperGetTensorPointerRW<float>(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  cudaEvent_t t_start, t_end;
  if (cc->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  backward_kernel(output_grad, input_grads, cc->numInputs, axis,
      out_grad_domain, in_grad_domains, stream);
  if (cc->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    printf("[%s] forward time = %.4f ms\n", cc->name, elapsed);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
  }
}

void Concat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(CONCAT_BWD_TASK_ID, task_is,
    TaskArgument(this, sizeof(Concat)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i].region_grad));
    //LogicalRegion lr = inputs[i].region_grad;
    //printf("concat[%d]: region(%d,%d,%d)\n", i+1, lr.get_index_space().get_id(), lr.get_field_space().get_id(), lr.get_tree_id());
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}


bool Concat::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics)
{
  assert (numInputs <= MAX_NUM_INPUTS);
  Tensor sub_inputs[MAX_NUM_INPUTS], sub_output;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  for (int i = 0; i < numInputs; i++) {
    if (!inputs[i].get_input_sub_tensor(pc, sub_inputs[i], op_type)) {
      return false;
    }
  }

  ConcatMeta *m = sim->concat_meta;
  this->init_meta(m);

  sim->free_all();
  float *input_ptrs[MAX_NUM_INPUTS];
  float *input_grad_ptrs[MAX_NUM_INPUTS];
  for (int i = 0; i < numInputs; i++) {
    input_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    assert (input_ptrs[i] != NULL);
  }
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  int axis = outputs[0].numDim - 1 - this->axis;

  Domain out_domain = sub_output.get_domain();
  Domain in_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < numInputs; i++) {
    in_domains[i] = sub_inputs[i].get_domain();
  }
  
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(output_ptr, input_ptrs, numInputs, axis, out_domain, in_domains, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    for (int i = 0; i < numInputs; i++) {
      input_grad_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
      assert (input_grad_ptrs[i] != NULL);
    }
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(output_grad_ptr, input_grad_ptrs,
        numInputs, axis, out_domain, in_domains, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Concat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Concat] name(%s) forward_time(%.4lf)\n",
        name, cost_metrics.forward_time);
  }

  return true;
}
