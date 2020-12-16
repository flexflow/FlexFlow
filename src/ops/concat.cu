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

Tensor FFModel::concat(int n, const Tensor* tensors,
                       int axis)
{
  Concat *cat = new Concat(*this, n, tensors, axis);
  layers.push_back(cat);
  return cat->outputs[0];
}

Concat::Concat(FFModel& model,
               int _n, const Tensor* _tensors,
               int _axis)
: Op(model, OP_CONCAT, "Concat_"+std::to_string(_axis), _n, _tensors), axis(_axis),
   profiling(model.config.profiling)
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

__host__
OpMeta* Concat::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  //FFHandler handler = *((const FFHandler*) task->local_args);
  //ConcatMeta* m = new ConcatMeta(handler);
  //return m;
  // Return null since Concat ops don't need ConcatMeta
  return NULL;
}

void Concat::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
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
  float *output;
  const float *inputs[MAX_NUM_INPUTS];
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(domain.get_dim() == cc->outputs[0].numDim);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      TensorAccessorW<float, DIM> accOutput( \
          regions[0], task->regions[0], FID_DATA, ctx, runtime, \
          false/*readOutput*/); \
      output = accOutput.ptr; \
      calc_blk_size<DIM>(num_blocks, output_blk_size, accOutput.rect, axis); \
      for (int i = 0; i < cc->numInputs; i++) { \
        TensorAccessorR<float, DIM> accInput( \
            regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime); \
        inputs[i] = accInput.ptr; \
        coord_t input_num_blocks = 1; \
        calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], accInput.rect, axis); \
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
  for (int i = 0; i < cc->numInputs; i++) {
    copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
        output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
    //printf("output = %x num_blocks=%d output_blk_size=%d input_blk_size[%d]=%d\n",
    //       output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
    output += input_blk_sizes[i];
  }
  if (cc->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    //print_tensor<4, float>(output - output_blk_size, output_rect, "[Concat:forward:output]");
    //printf("output_blk_size=%zu\n", output_blk_size);
    //print_tensor<4, float>(inputs[0], input_rect[0], "[Concat:forward:input0]");
    //print_tensor<4, float>(inputs[1], input_rect[1], "[Concat:forward:input1]");
  }
#ifdef DEADCODE
  const AccessorWO<float, 3> acc_output(regions[0], FID_DATA);
  Rect<3> rect_output;
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  float *output_ptr = acc_output.ptr(rect_output.lo);
  float *output_bound = output_ptr + rect_output.volume();
  for (int i = 0; i < cc->numInputs; i++) {
    const AccessorRO<float, 3> acc_input(regions[i+1], FID_DATA);
    Rect<3> rect_input =
      runtime->get_index_space_domain(ctx, task->regions[i+1].region.get_index_space());
    assert(acc_input.accessor.is_dense_arbitrary(rect_input));
    const float *input_ptr = acc_input.ptr(rect_input.lo);
    checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
                              rect_input.volume() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    output_ptr += rect_input.volume();
  }
  assert(output_ptr == output_bound);
#endif
}

void Concat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
#ifdef DEADCODE
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
#endif
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
  const float *output_grad;
  float *input_grads[MAX_NUM_INPUTS];
  coord_t num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(domain.get_dim() == cc->outputs[0].numDim);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      TensorAccessorR<float, DIM> accOutputGrad( \
          regions[0], task->regions[0], FID_DATA, ctx, runtime); \
      output_grad = accOutputGrad.ptr; \
      calc_blk_size<DIM>(num_blocks, output_blk_size, accOutputGrad.rect, axis); \
      for (int i = 0; i < cc->numInputs; i++) { \
        TensorAccessorW<float, DIM> accInputGrad( \
            regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime, \
            true/*readOutput*/); \
        input_grads[i] = accInputGrad.ptr; \
        coord_t input_num_blocks = 1; \
        calc_blk_size<DIM>(input_num_blocks, input_blk_sizes[i], accInputGrad.rect, axis); \
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
  for (int i = 0; i < cc->numInputs; i++) {
    add_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
        input_grads[i], output_grad, num_blocks, input_blk_sizes[i], output_blk_size);
    output_grad += input_blk_sizes[i];
  }
  if (cc->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    int batch_size = domain.get_volume() / output_blk_size;
    Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size - 1));
    Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1, batch_size - 1));
    //print_tensor<2, float>(output_grad - output_blk_size, output_rect, "[Concat:backward:output]");
    //print_tensor<2, float>(input_grads[0], input_rect, "[Concat:backward:input0]");
  }
#ifdef DEADCODE
  const AccessorRO<float, 3> acc_output(regions[0], FID_DATA);
  Rect<3> rect_output;
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  float *output_ptr = (float*) acc_output.ptr(rect_output.lo);
  float *output_bound = output_ptr + rect_output.volume();
  for (int i = 0; i < cc->numInputs; i++) {
    const AccessorWO<float, 3> acc_input(regions[i+1], FID_DATA);
    Rect<3> rect_input =
      runtime->get_index_space_domain(ctx, task->regions[i+1].region.get_index_space());
    assert(acc_input.accessor.is_dense_arbitrary(rect_input));
    float *input_ptr = acc_input.ptr(rect_input.lo);
    checkCUDA(cudaMemcpyAsync(input_ptr, output_ptr,
                              rect_input.volume() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    output_ptr += rect_input.volume();
  }
  assert(output_ptr == output_bound);
#endif
}

void Concat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
#ifdef DEADCODE
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
#endif
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


bool Concat::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  //TODO: implement measure_forward
  forward_time = 0.0f;
  backward_time = 0.0f;
  return true;
}
