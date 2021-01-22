/* Copyright 2019 Stanford, NVIDIA
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

Tensor FFModel::dense(const Tensor& input,
                      int outDim,
                      ActiMode activation,
                      bool use_bias,
                      const Op* shared_op,
                      Initializer* kernel_initializer,
                      Initializer* bias_initializer,
                      const char *name)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }
  Linear *li = new Linear(*this, input, outDim, activation, use_bias,
                          shared_op, kernel_initializer, bias_initializer, name);
  layers.push_back(li);
  return li->outputs[0];
}

Linear* FFModel::dense(int inDim, int outDim,
                       ActiMode activation,
                       bool use_bias,
                       Initializer* kernel_initializer,
                       Initializer* bias_initializer,
                       const char *name)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }
  Linear *li = new Linear(*this, inDim, outDim, activation, use_bias,
                          kernel_initializer, bias_initializer, name);
  layers.push_back(li);
  return li;
}

Linear::Linear(FFModel& model,
               const Tensor& _input,
               int out_dim,
               ActiMode _activation,
               bool _use_bias,
               const Op* shared_op,
               Initializer* _kernel_initializer,
               Initializer* _bias_initializer,
               const char* name)
: Op(model, OP_LINEAR, shared_op, name, _input),
  in_channels(_input.adim[0]), out_channels(out_dim),
  activation(_activation), use_bias(_use_bias),
  kernel_initializer(_kernel_initializer),
  bias_initializer(_bias_initializer),
  profiling(model.config.profiling)
{
  numInputs = 1;
  numOutputs = 1;
  outputs[0].numDim = _input.numDim;
  for (int i = 1; i < outputs[0].numDim; i++)
    outputs[0].adim[i] = _input.adim[i];
  outputs[0].adim[0] = out_dim;
  weights[0].numDim = 2;
  weights[0].adim[0] = in_channels;
  weights[0].adim[1] = out_channels;
  numWeights = 1;
  if (use_bias) {
    weights[1].numDim = 1;
    weights[1].adim[0] = out_channels;
    numWeights = 2;
  }
}

Linear::Linear(FFModel& model,
               int in_dim, int out_dim,
               ActiMode _activation,
               bool _use_bias,
               Initializer* _kernel_initializer,
               Initializer* _bias_initializer,
               const char* name)
: Op(model, OP_LINEAR, name, 1),
  in_channels(in_dim), out_channels(out_dim),
  activation(_activation), use_bias(_use_bias),
  kernel_initializer(_kernel_initializer),
  bias_initializer(_bias_initializer),
  profiling(model.config.profiling)
{
}

Tensor Linear::init_inout(FFModel& model, const Tensor& _input)
{
  assert(_input.adim[0] == in_channels);
  inputs[0] = _input;
  create_output_and_partition(model);
  return outputs[0];
}

/*
void Linear::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
  model.parameters.push_back(weights[0]);
  if (numWeights > 1) { // bias is used
    assert(numWeights == 2);
    model.parameters.push_back(weights[1]);
  }
}
*/


void Linear::create_weights(FFModel& model)
{
  int dim = inputs[0].numDim;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      create_weights_with_dim<DIM>(model); \
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
void Linear::create_weights_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, pcname));

#ifdef FF_ENABLE_NCCL
  Parameter::CommType comm_type = Parameter::NCCL;
#else
  Parameter::CommType comm_type = Parameter::PS;
#endif

  // Create kernel tensor
  {
    const int dims[2] = {out_channels, in_channels};
    weights[0] = model.create_linear_weight<2, NDIM>(this, dims, DT_FLOAT,
        kernel_initializer, true/*create_grad*/, comm_type);
  }
  // Create bias tensor
  if (use_bias) {
    const int dims[1] = {out_channels};
    weights[1] = model.create_linear_weight<1, NDIM>(this, dims, DT_FLOAT,
        bias_initializer, true/*create_grad*/, comm_type);
    assert(numWeights == 2);
  } else {
    assert(numWeights == 1);
  }
}

void Linear::create_output_and_partition(FFModel& model)
{
  int dim = inputs[0].numDim;
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
      // Unsupported dim for ElementWiseBinary operator
      assert(false);
    }
  }
}

template<int NDIM>
void Linear::create_output_and_partition_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_n = part_rect.hi[NDIM-1] - part_rect.lo[NDIM-1] + 1;
  int in_dim = inputs[0].adim[0];
  assert(in_dim == in_channels);
  int batch_size = inputs[0].adim[NDIM-1];
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
  // Create replica tensor
  if (num_par_c > 1) {
    const int dims[3] = {num_par_c, batch_size, in_dim};
    replica = model.create_linear_replica<3>(dims, (IndexSpaceT<NDIM>)task_is, DT_FLOAT);
    {
      Rect<NDIM> extent;
      for (int i = 0; i < NDIM; i++) {
        extent.lo[i] = 0;
        assert(outputs[0].adim[i] % (part_rect.hi[i] - part_rect.lo[i] + 1) == 0);
        extent.hi[i] = outputs[0].adim[i] / (part_rect.hi[i] - part_rect.lo[i] + 1) - 1;
      }
      Transform<NDIM, NDIM> transform;
      for (int i = 0; i < NDIM; i++)
        for (int j = 0; j < NDIM; j++)
          transform[i][j] = 0;
      for (int i = 1; i < NDIM; i++)
        transform[i][i] = extent.hi[i] + 1;
      IndexPartition ip = runtime->create_partition_by_restriction(
          ctx, inputs[0].region.get_index_space(), task_is, transform, extent);
      input_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region, ip);
    }
    // Backward use the same ip as inputs[0]
    input_grad_lps[0] = inputs[0].part_grad;
    {
      IndexSpaceT<NDIM> input_task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(input_rect));
      Rect<NDIM+1> extent;
      for (int i = 0; i < NDIM; i++) {
        extent.lo[i] = 0;
        assert(inputs[0].adim[i] % (input_rect.hi[i] - input_rect.lo[i] + 1) == 0);
        extent.hi[i] = inputs[0].adim[i] / (input_rect.hi[i] - input_rect.lo[i] + 1) - 1;
      }
      extent.lo[NDIM] = 0;
      extent.hi[NDIM] = num_par_c - 1;
      Transform<NDIM+1, NDIM> transform;
      for (int i = 0; i < NDIM+1; i++)
        for (int j = 0; j < NDIM; j++)
          transform[i][j] = 0;
      for (int i = 0; i < NDIM; i++)
        transform[i][i] = inputs[0].adim[i] / (input_rect.hi[i] - input_rect.lo[i] + 1);
      IndexPartition ip = runtime->create_partition_by_restriction(
          ctx, replica.region_grad.get_index_space(), input_task_is,
          transform, extent);
      assert(runtime->is_index_partition_disjoint(ctx, ip));
      assert(runtime->is_index_partition_complete(ctx, ip));
      // Note we use replica.part to save how to partition the replica
      // to compute input_grad_lps
      replica.part = runtime->get_logical_partition(
          ctx, replica.region_grad, ip);
    }
  } else {
    if (input_rect == part_rect) {
      input_lps[0] = inputs[0].part;
      input_grad_lps[0] = inputs[0].part_grad;
    } else {
      Rect<NDIM> extent;
      for (int i = 0; i < NDIM; i++) {
        extent.lo[i] = 0;
        assert(inputs[0].adim[i] % (part_rect.hi[i] - part_rect.lo[i] + 1) == 0);
        extent.hi[i] = inputs[0].adim[i] / (part_rect.hi[i] - part_rect.lo[i] + 1) - 1;
      }
      Transform<NDIM, NDIM> transform;
      for (int i = 0; i < NDIM; i++)
        for (int j = 0; j < NDIM; j++) {
          transform[i][j] = 0;
          if (i==j)
            transform[i][j] = extent.hi[i] + 1;
        }
      IndexPartition ip = runtime->create_partition_by_restriction(
          ctx, inputs[0].region.get_index_space(), task_is, transform, extent);
      assert(runtime->is_index_partition_disjoint(ctx, ip));
      assert(runtime->is_index_partition_complete(ctx, ip));
      input_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region, ip);
      input_grad_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region_grad, ip);
    }
  }
}

/*
  regions[0](O): output
  regions[1](I): kernel
  regions[2](I): bias
*/
OpMeta* Linear::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime* runtime)
{
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (out_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return init_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  return NULL;
}

template<int NDIM>
OpMeta* Linear::init_task_with_dim(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Linear* linear = (Linear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  //TensorAccessorR<float, 2> acc_input(
  //    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  //int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  //printf("init linear (input): in_dim(%d) out_dim(%d) batch_size(%d)\n",
  //    in_dim, out_dim, batch_size);
  LinearMeta* m = new LinearMeta(handle, batch_size);
  m->activation = linear->activation;

  if (m->activation != AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (linear->activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch_size, out_dim, 1, 1));
  }
#ifdef FF_ENABLE_NCCL
  m->init_nccl_communicator(task, linear->ncclId);
#endif
  return m;
}

void Linear::init(const FFModel& ff)
{
  int dim = outputs[0].numDim;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
      return init_with_dim<DIM>(ff);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
void Linear::init_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  ParallelConfig pc;
  std::string pcname = name;
  ff.config.find_parallel_config(NDIM, pcname, pc);
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[pc.device_ids[idx++]];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(LINEAR_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  //launcher.add_region_requirement(
  //    RegionRequirement(input_lps[0], 0/*projection id*/,
  //                      READ_ONLY, EXCLUSIVE, inputs[0].region));
  //launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[1].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[1].region));
  launcher.add_field(2, FID_DATA);
  // Add inputs[0].region_grad to avoid Legion warning
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*static*/
void Linear::forward_kernel(const LinearMeta* m,
                            const float* input_ptr,
                            float* output_ptr,
                            const float* kernel_ptr,
                            const float* bias_ptr,
                            int in_dim, int out_dim, int batch_size)
{
  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        out_dim, batch_size, in_dim,
                        &alpha, kernel_ptr, in_dim,
                        input_ptr, in_dim, &beta,
                        output_ptr, out_dim));
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        out_dim, batch_size, 1,
                        &alpha, bias_ptr, 1,
                        m->one_ptr, 1, &alpha,
                        output_ptr, out_dim));
  if (m->activation != AC_MODE_NONE) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr,
        &beta, m->outputTensor, output_ptr));
  }
}

__host__
void Linear::forward_task(const Task *task,
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
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
  regions[3](I): bias
*/
template<int NDIM>
void Linear::forward_task_with_dim(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  TensorAccessorR<float, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_input.rect.volume() == in_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  assert(acc_bias.rect.volume() == out_dim);

  cudaEvent_t t_start, t_end;
  if (linear->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  linear->forward_kernel(m, acc_input.ptr, acc_output.ptr,
      acc_kernel.ptr, acc_bias.ptr, in_dim, out_dim, batch_size);

  if (linear->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Linear] forward time = %.2lfms\n", linear->name, elapsed);
    //print_tensor<NDIM, float>(acc_input.ptr, acc_input.rect, "[Linear:forward:input]");
    //print_tensor<2, float>(acc_kernel.ptr, acc_kernel.rect, "[Linear:forward:kernel]");
    //print_tensor<1, float>(acc_bias.ptr, acc_bias.rect, "[Linear:forward:bias]");
    //print_tensor<NDIM, float>(acc_output.ptr, acc_output.rect, "[Linear:forward:output]");
  }
}

void Linear::forward(const FFModel& ff)
{
  int dim = outputs[0].numDim;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
      return forward_with_dim<DIM>(ff);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
void Linear::forward_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(LINEAR_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[1].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[1].region));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

__global__
void sigmoid_backward(float *grad_ptr, const float *output, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = grad_ptr[i] * output[i] * (1 - output[i]);
  }
}

/*static*/
void Linear::backward_kernel(const LinearMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             float* output_grad_ptr,
                             const float* kernel_ptr,
                             float* kernel_grad_ptr,
                             float* bias_grad_ptr,
                             int in_dim, int out_dim, int batch_size)
{
  float alpha = 1.0f;
  int output_size = out_dim * batch_size;
  if (m->activation == AC_MODE_RELU) {
    reluBackward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
        output_grad_ptr, output_ptr, output_size);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
        output_grad_ptr, output_ptr, output_size);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        in_dim, out_dim, batch_size,
                        &alpha, input_ptr, in_dim,
                        output_grad_ptr, out_dim,
                        &alpha, kernel_grad_ptr, in_dim));
  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  checkCUDA(cublasSgemv(m->handle.blas, CUBLAS_OP_N,
                        out_dim, batch_size,
                        &alpha, output_grad_ptr, out_dim,
                        m->one_ptr, 1,
                        &alpha, bias_grad_ptr, 1));
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        in_dim, batch_size, out_dim,
                        &alpha, kernel_ptr, in_dim,
                        output_grad_ptr, out_dim,
                        &alpha, input_grad_ptr, in_dim));
}

void Linear::backward_task(const Task *task,
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
  regions[0](I): input
  regions[1](I/O): replica_grad or input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
template<int NDIM>
__host__
void Linear::backward_task_with_dim(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  float* input_grad = NULL;
  TensorAccessorR<float, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, NDIM> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  if (domain.get_dim() == NDIM+1) {
    assert(false);
    TensorAccessorW<float, NDIM+1> acc_replica_grad(
        regions[1], task->regions[1], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    assert(acc_replica_grad.rect.volume() == in_dim * batch_size);
    input_grad = acc_replica_grad.ptr;
  } else {
    TensorAccessorW<float, NDIM> acc_replica_grad(
        regions[1], task->regions[1], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    assert(acc_replica_grad.rect.volume() == in_dim * batch_size);
    input_grad = acc_replica_grad.ptr;
  }
  TensorAccessorW<float, NDIM> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_kernel_grad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorW<float, 1> acc_bias_grad(
      regions[6], task->regions[6], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  // make sure the sizes match
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_output_grad.rect.volume() == out_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  assert(acc_kernel_grad.rect.volume() == in_dim * out_dim);
  assert(acc_bias_grad.rect.volume() == out_dim);
  cudaEvent_t t_start, t_end;
  if (linear->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  linear->backward_kernel(m, acc_input.ptr, input_grad,
      acc_output.ptr, acc_output_grad.ptr,
      acc_kernel.ptr, acc_kernel_grad.ptr,
      acc_bias_grad.ptr, in_dim, out_dim, batch_size);
  if (linear->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear backward time = %.2lfms\n", elapsed);
    //print_tensor<NDIM, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Linear:backward:output_grad]");
    //print_tensor<2, float>(acc_kernel_grad.ptr, acc_kernel_grad.rect, "[Linear:backward:kernel_grad]");
    //print_tensor<1, float>(acc_bias_grad.ptr, acc_bias_grad.rect, "[Linear:backward:bias_grad]");
    //print_tensor<2, float>(input_grad, acc_input.rect, "[Linear:backward:input_grad]");
  }
}

void Linear::backward2_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward2_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}


/*
  regions[0](I/O): input_grad
  regions[1](I): replicas
*/
template<int NDIM>
__host__
void Linear::backward2_task_with_dim(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
  float alpha = 1.0f;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  TensorAccessorW<float, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 3> acc_replica(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input.rect.hi[0] == acc_replica.rect.hi[0]);
  assert(acc_input.rect.lo[0] == acc_replica.rect.lo[0]);
  assert(acc_input.rect.hi[1] == acc_replica.rect.hi[1]);
  assert(acc_input.rect.lo[1] == acc_replica.rect.lo[1]);
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  int num_replica = acc_replica.rect.hi[NDIM] - acc_replica.rect.lo[NDIM] + 1;
  const float *replica_ptr = acc_replica.ptr;
  for (int i = 1; i < num_replica; i++) {
    checkCUDA(cublasSaxpy(m->handle.blas, acc_input.rect.volume(),
                          &alpha, replica_ptr, 1, acc_input.ptr, 1));
    replica_ptr += acc_input.rect.volume();
  }
}

void Linear::backward(const FFModel& ff)
{
  int dim = outputs[0].numDim;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward_with_dim<DIM>(ff);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
void Linear::backward_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  {
    IndexLauncher launcher(LINEAR_BWD_TASK_ID, task_is,
                           TaskArgument(this, sizeof(Linear)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string(name)));
    // regions[0](I): input
    launcher.add_region_requirement(
        RegionRequirement(input_lps[0], 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[0].region));
    launcher.add_field(0, FID_DATA);
    // regions[1](I/O): replica_grad
    if (replica.region_grad != LogicalRegion::NO_REGION) {
      launcher.add_region_requirement(
          RegionRequirement(replica.part_grad, 0/*projection id*/,
                            READ_WRITE, EXCLUSIVE, replica.region_grad));
      launcher.add_field(1, FID_DATA);
    } else {
      launcher.add_region_requirement(
          RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                            READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
      launcher.add_field(1, FID_DATA);
    }
    // regions[2](I): output
    launcher.add_region_requirement(
        RegionRequirement(outputs[0].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, outputs[0].region));
    launcher.add_field(2, FID_DATA);
    // regions[3](I/O): output_grad
    launcher.add_region_requirement(
        RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, outputs[0].region_grad));
    launcher.add_field(3, FID_DATA);
    // regions[4](I): filter
    launcher.add_region_requirement(
        RegionRequirement(weights[0].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[0].region));
    launcher.add_field(4, FID_DATA);
    // regions[5](I/O): filter_grad
    launcher.add_region_requirement(
        RegionRequirement(weights[0].part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, weights[0].region_grad));
    launcher.add_field(5, FID_DATA);
    // regions[6](I/O): bias_grad
    launcher.add_region_requirement(
        RegionRequirement(weights[1].part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, weights[1].region_grad));
    launcher.add_field(6, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  if (replica.region_grad != LogicalRegion::NO_REGION) {
    // We aggregate parameters from replica tensor to input tensor
    // Note we use input's task_is to reduce extra data transfers
    Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part_grad.get_index_partition());
    IndexSpaceT<2> input_task_is = IndexSpaceT<2>(ff.get_task_is(input_rect));
    IndexLauncher launcher(LINEAR_BWD2_TASK_ID, task_is,
                           TaskArgument(this, sizeof(Linear)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string(name)));
    launcher.add_region_requirement(
        RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
    launcher.add_field(0, FID_DATA);
    // Note that replica.part save's a partition of replica.region_grad
    launcher.add_region_requirement(
        RegionRequirement(replica.part, 0/*partition id*/,
                          READ_ONLY, EXCLUSIVE, replica.region_grad));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

/*
__host__
Parameter* Linear::get_parameter(int index)
{
  if (index == 0) {
    return &weights[0];
  } else if (index == 1){
    return &weights[1];
  } else {
    assert(0);
    return NULL;
  }
}
*/

__host__
void Linear::print_layer(const FFModel& ff)
{
  printf("linear layer\n");
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  RegionRequirement kernel_req(weights[0].region, READ_WRITE, EXCLUSIVE, weights[0].region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

  RegionRequirement bias_req(weights[1].region, READ_WRITE, EXCLUSIVE, weights[1].region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();

  TensorAccessorW<float, 2> acc_kernel(kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
  TensorAccessorW<float, 1> acc_bias(bias_region, bias_req, FID_DATA, ctx, runtime, true);

  const float *kernel_ptr = acc_kernel.ptr;
  const float *bias_ptr = acc_bias.ptr;

  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  size_t bias_size = acc_bias.rect.volume();
  printf("kernel, %p, %d, [%d, %d]\n", kernel_ptr, kernel_size, kernel_dim1, kernel_dim2);
  printf("bias, %p, %d\n", bias_ptr, bias_size);

  for (int i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");

  for (int i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");

  runtime->unmap_region(ctx, kernel_region);
  runtime->unmap_region(ctx, bias_region);

}

LinearMeta::LinearMeta(FFHandler handler, int batch_size)
: OpMeta(handler)
{
  // Allocate an all-one's vector
  float* dram_one_ptr = (float *) malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++)
    dram_one_ptr[i] = 1.0f;
  float* fb_one_ptr;
  checkCUDA(cudaMalloc(&fb_one_ptr, sizeof(float) * batch_size));
  checkCUDA(cudaMemcpy(fb_one_ptr, dram_one_ptr,
                       sizeof(float) * batch_size, cudaMemcpyHostToDevice));
  one_ptr = (const float*) fb_one_ptr;
  // Allocate descriptors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}

bool Linear::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  Tensor sub_output, sub_input;
  if (!outputs[0].get_output_sub_tensor(pc, sub_output, OP_LINEAR))
    return false;
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, OP_LINEAR))
    return false;
  int input_c = sub_input.adim[0];
  int input_n = sub_input.get_volume() / input_c;
  int output_c = sub_output.adim[0];
  int output_n = sub_output.get_volume() / output_c;
  LinearMeta* m = sim->linear_meta;
  m->activation = activation;
  if (activation != AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, 1, 1));
  }
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float* kernel_ptr = (float*)sim->allocate((size_t)output_c * input_c, DT_FLOAT);
  assert(kernel_ptr != NULL);
  float* bias_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);

  auto forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, kernel_ptr, bias_ptr,
        input_c, output_c, input_n);
  };
  auto backward = [&] {
    backward_kernel(m, input_ptr, input_ptr, output_ptr, output_ptr,
        kernel_ptr, kernel_ptr, bias_ptr, input_c, output_c, input_n);
  };

  inner_measure_compute_time(sim, forward, backward, forward_time, backward_time);

  printf("[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
         name, input_n, input_c, output_n, output_c, forward_time, backward_time);
  return true;
}

ParallelConfig Linear::get_random_parallel_config(const FFModel& ff) const
{
  if (!ff.config.enable_parameter_parallel)
    return Op::get_random_parallel_config(ff);
  std::vector<int> batch_candidates;
  std::vector<int> channel_candidates;
  int batch = outputs[0].adim[outputs[0].numDim-1];
  int channel = outputs[0].adim[0];
  int total_devices = ff.config.workersPerNode * ff.config.numNodes;
  for (int i = 1; i <= ff.config.workersPerNode; i++)
    if (channel % i == 0)
      for (int j = 1; i * j <= total_devices; j++)
        if (batch % j == 0) {
          batch_candidates.push_back(j);
          channel_candidates.push_back(i);
        }
  assert(batch_candidates.size() > 0);
  int idx = std::rand() % batch_candidates.size();
  int num_par_c = channel_candidates[idx];
  int num_par_b = batch_candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0].numDim;
  pc.dim[0] = num_par_c;
  pc.dim[pc.nDims-1] = num_par_b;
  for (int i = 1; i < pc.nDims - 1; i++)
    pc.dim[i] = 1;
  int start_idx = std::rand() % (total_devices - num_par_c * num_par_b + 1);
  start_idx = start_idx - start_idx % num_par_c;
  for (int i = 0; i < num_par_c * num_par_b; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

