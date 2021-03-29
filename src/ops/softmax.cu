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

using namespace Legion;

Tensor FFModel::softmax(const Tensor _input, int dim, const char *name)
{
  if (dim < 0)
    dim += _input->num_dims;
  Softmax *sm = new Softmax(*this, _input, _input->num_dims-1-dim, name);
  layers.push_back(sm);
  return sm->outputs[0];
}

SoftmaxMeta::SoftmaxMeta(FFHandler handler,
                         const Softmax* softmax,
                         const Domain& input_domain)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, input_domain));
  dim = softmax->dim;
  profiling = softmax->profiling;
  std::strcpy(op_name, softmax->name);
}

Softmax::Softmax(FFModel& model,
                 const Tensor _input,
                 int _dim,
                 const char* name)
: Op(model, OP_SOFTMAX, name, 1/*inputs*/, 0/*weights*/, _input),
  dim(_dim)
{
  // Currently assume we always perform softmax along the inner most dim
  assert(dim == 0);
  ParallelDim dims[MAX_TENSOR_DIM];
  int numdim = _input->num_dims;
  for (int i = 0; i < numdim; i++)
    dims[i] = _input->dims[numdim-1-i];
  outputs[0] = model.create_tensor(numdim, dims, DT_FLOAT, this);
}

void Softmax::create_input_partition(FFModel& model)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      create_output_and_partition_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim
      assert(false);
    }
  }
}

template<int NDIM>
void Softmax::create_output_and_partition_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  //int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  // Current require data parallelism for Softmax
  assert(num_par_c == 1);
  return Op::create_input_partition(model);
#ifdef DEADCODE
  {
    int dims[NDIM];
    for (int i = 0; i < NDIM; i++)
      dims[i] = outputs[0].adim[NDIM-1-i];
    outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  // Compute partition bound for input
  Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0]->part;
    input_grad_lps[0] = inputs[0]->part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], (IndexSpaceT<NDIM>)task_is, input_lps[0], input_grad_lps[0]);
  }
#endif
}

/*
  regions[0]: input
  regions[1]: output
 */
OpMeta* Softmax::init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Softmax* softmax = (Softmax*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(input_domain == output_domain);
  SoftmaxMeta* m = new SoftmaxMeta(handle, softmax, output_domain);
  //checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  return m;
}

__host__
void Softmax::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(SOFTMAX_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Softmax)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/* static */
void Softmax::forward_kernel(SoftmaxMeta const *m,
                             float const *input_ptr,
                             float *output_ptr)
{
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->inputTensor, output_ptr));
}

void Softmax::forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return forward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/
template<int NDIM>
__host__
void Softmax::forward_task_with_dim(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Softmax* softmax = (Softmax*) task->args;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  TensorAccessorR<float, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  forward_kernel(m, acc_input.ptr, acc_output.ptr);
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<2, float>(acc_input.ptr, acc_input.rect, "[Softmax:forward:input]");
    //print_tensor<2, float>(acc_output.ptr, acc_output.rect, "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Softmax] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

__host__
void Softmax::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(SOFTMAX_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/* static */
void Softmax::backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements)
{
  checkCUDA(cudaMemcpyAsync(input_grad_ptr, output_grad_ptr,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
}

void Softmax::backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I/O): input_grad
  regions[1](I): output_grad
*/
// Note that the backward task of softmax is actually a no op (i.e., input_grad = output_grad)
// since the upstream cross_entropy_loss function computes performs softmax_cross_entropy_loss
// to avoid intermediate zeros
template<int NDIM>
__host__
void Softmax::backward_task_with_dim(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Softmax* softmax = (Softmax*) task->args;
  const SoftmaxMeta* m = *((SoftmaxMeta**) task->local_args);
  TensorAccessorW<float, NDIM> acc_input_grad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, NDIM> acc_output_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // make sure the image indices match!
  assert(acc_input_grad.rect == acc_output_grad.rect);

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  //checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  backward_kernel(acc_input_grad.ptr, acc_output_grad.ptr, acc_input_grad.rect.volume());
  if (m->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<2, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Softmax:backward:output_grad]");
    //print_tensor<2, float>(acc_input_grad.ptr, acc_input_grad.rect, "[Softmax:backward:input_grad]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Softmax backward time = %.2fms\n", elapsed);
  }
}

__host__
void Softmax::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Softmax::measure_operator_cost(Simulator* sim,
                                    const ParallelConfig& pc,
                                    CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_input;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }

  SoftmaxMeta *m = new SoftmaxMeta(sim->handler, this, sub_output.get_domain());

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert (input_ptr != NULL);
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert (output_ptr != NULL);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert (output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(input_grad_ptr, output_grad_ptr, sub_output.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Softmax] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Softmax] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
        name, sub_output.get_volume(),
        cost_metrics.forward_time);
  }
  // Free softmaxmeta
  delete m;
  return true;
}

bool Softmax::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_SOFTMAX_DIM:
      *value = dim;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

Node FFModel::get_or_create_softmax_node(const Tensor input,
                                         int softmax_dim)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(softmax_dim);
  const auto& it = cached_softmax_ops.find(hash);
  Softmax* softmax = NULL;
  if (it != cached_softmax_ops.end()) {
    softmax = it->second;
  } else {
    softmax = new Softmax(*this, input, softmax_dim, NULL);
    cached_softmax_ops[hash] = softmax;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = softmax;
  return ret;
}
