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
                      Initializer* kernel_initializer,
                      Initializer* bias_initializer)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }
  Linear *li = new Linear(*this, input, outDim, activation, use_bias,
                          kernel_initializer, bias_initializer);
  layers.push_back(li);
  return li->outputs[0];
}

Linear* FFModel::dense(int inDim, int outDim, 
                       ActiMode activation,
                       bool use_bias, 
                       Initializer* kernel_initializer,
                       Initializer* bias_initializer)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }
  Linear *li = new Linear(*this, inDim, outDim, activation, use_bias,
                          kernel_initializer, bias_initializer);
  layers.push_back(li);
  return li;
}

Linear::Linear(FFModel& model,
               const Tensor& _input,
               int out_dim,
               ActiMode _activation,
               bool _use_bias,
               Initializer* _kernel_initializer,
               Initializer* _bias_initializer)
: Op(model, OP_LINEAR, "Dense_"+std::to_string(out_dim), _input), 
  in_channels(_input.adim[0]), out_channels(out_dim),
  activation(_activation), use_bias(_use_bias),
  kernel_initializer(_kernel_initializer),
  bias_initializer(_bias_initializer),
  profiling(model.config.profiling)
{
  assert(_input.numDim == 2);
  int batch_size = _input.adim[1];
  outputs[0].numDim = 2;
  outputs[0].adim[0] = out_channels;
  outputs[0].adim[1] = batch_size;
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
               Initializer* _bias_initializer)
: Op(model, OP_LINEAR, "Dense_"+std::to_string(out_dim), 1), 
  in_channels(in_dim), out_channels(out_dim),
  activation(_activation), use_bias(_use_bias),
  kernel_initializer(_kernel_initializer),
  bias_initializer(_bias_initializer),
  profiling(model.config.profiling)
{
}

Tensor Linear::init_inout(FFModel& model, const Tensor& _input)
{
  assert(_input.numDim == 2);
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
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));

  // Create kernel tensor
  {
    const int dims[2] = {out_channels, in_channels};
    weights[0] = model.create_linear_weight<2>(this, dims, (IndexSpaceT<2>)task_is, DT_FLOAT, kernel_initializer);
  }
  // Create bias tensor
  if (use_bias) {
    const int dims[1] = {out_channels};
    weights[1] = model.create_linear_weight<1>(this, dims, (IndexSpaceT<2>)task_is, DT_FLOAT, bias_initializer);
    assert(numWeights == 2);
  } else {
    assert(numWeights == 1);
  }
}

void Linear::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  int in_dim = inputs[0].adim[0];
  assert(in_dim == in_channels);
  int batch_size = inputs[0].adim[1];
  {
    const int dims[2] = {batch_size, out_channels};
    outputs[0] = model.create_tensor<2>(dims, (IndexSpaceT<2>)task_is, DT_FLOAT);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }
  // Compute partition bound for input
  Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  // Create replica tensor
  if (num_par_c > 1) {
    const int dims[3] = {num_par_c, batch_size, in_dim};
    replica = model.create_linear_replica<3>(dims, (IndexSpaceT<2>)task_is, DT_FLOAT);
    {
      Rect<2> extent(Point<2>(0, 0), Point<2>(in_dim-1, batch_size/num_par_n-1));
      Transform<2, 2> transform;
      transform[0][0] = 0;
      transform[0][1] = 0;
      transform[1][0] = 0;
      transform[1][1] = batch_size/num_par_n;
      IndexPartition ip = runtime->create_partition_by_restriction(
          ctx, inputs[0].region.get_index_space(), task_is, transform, extent);
      input_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region, ip);
    }
    // Backward use the same ip as inputs[0]
    input_grad_lps[0] = inputs[0].part_grad;
    {
      IndexSpaceT<2> input_task_is = IndexSpaceT<2>(model.get_or_create_task_is(input_rect));
      const coord_t num_parts[2] = {input_rect.hi[0] - input_rect.lo[0] + 1,
                                    input_rect.hi[1] - input_rect.lo[1] + 1};
      Rect<3> extent(Point<3>(0, 0, 0),
          Point<3>(in_dim/num_parts[0]-1, batch_size/num_parts[1]-1, num_par_c-1));
      Transform<3, 2> transform;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
          transform[i][j] = 0;
      transform[0][0] = in_dim / num_parts[0];
      transform[1][1] = batch_size / num_parts[1];
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
      Rect<2> extent(Point<2>(0,0), Point<2>(in_dim-1,batch_size/num_par_n-1));
      Transform<2, 2> transform;
      transform[0][0] = 0;
      transform[0][1] = 0;
      transform[1][0] = 0;
      transform[1][1] = batch_size / num_par_n;
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
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Linear* linear = (Linear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  //TensorAccessorR<float, 2> acc_input(
  //    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  //int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  printf("init linear (input): in_dim(%d) out_dim(%d) batch_size(%d)\n",
      in_dim, out_dim, batch_size);
  LinearMeta* m = new LinearMeta(handle, batch_size);

  if (linear->activation != AC_MODE_NONE) {
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
  return m;
}

void Linear::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

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
  if (activation != AC_MODE_NONE) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr,
        &beta, m->outputTensor, output_ptr));
  }
}

/*
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
  regions[3](I): bias
*/
__host__
void Linear::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  TensorAccessorR<float, 2> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 1> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  assert(acc_output.rect.volume() == out_dim * batch_size);
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
    printf("Linear forward time = %.2lfms\n", elapsed);
    //print_tensor<2, float>(acc_input.ptr, acc_input.rect, "[Linear:forward:input]");
    //print_tensor<2, float>(acc_kernel.ptr, acc_kernel.rect, "[Linear:forward:kernel]");
    //print_tensor<1, float>(acc_bias.ptr, acc_bias.rect, "[Linear:forward:bias]");
    //print_tensor<2, float>(acc_output.ptr, acc_output.rect, "[Linear:forward:output]");
  }
}

void Linear::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
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

void Linear::backward_kernel(const LinearMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             float* output_grad_ptr,
                             const float* kernel_ptr,
                             float* kernel_grad_ptr,
                             float* bias_grad_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size)
{
  float alpha = 1.0f;
  int output_size = out_dim * batch_size;
  if (activation == AC_MODE_RELU) {
    reluBackward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
        output_grad_ptr, output_ptr, output_size);
  } else if (activation == AC_MODE_SIGMOID) {
    sigmoid_backward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
        output_grad_ptr, output_ptr, output_size);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(activation == AC_MODE_NONE);
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

/*
  regions[0](I): input
  regions[1](I/O): replica_grad or input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
__host__
void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  float* input_grad = NULL;
  TensorAccessorR<float, 2> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 2> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int batch_size = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  if (domain.get_dim() == 3) {
    TensorAccessorW<float, 3> acc_replica_grad(
        regions[1], task->regions[1], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    assert(acc_replica_grad.rect.volume() == in_dim * batch_size);
    input_grad = acc_replica_grad.ptr;
  } else {
    TensorAccessorW<float, 2> acc_replica_grad(
        regions[1], task->regions[1], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    assert(acc_replica_grad.rect.volume() == in_dim * batch_size);
    input_grad = acc_replica_grad.ptr;
  }
  TensorAccessorW<float, 2> acc_output_grad(
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
    //print_tensor<2, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Linear:backward:output_grad]");
    //print_tensor<2, float>(acc_kernel_grad.ptr, acc_kernel_grad.rect, "[Linear:backward:kernel_grad]");
    //print_tensor<1, float>(acc_bias_grad.ptr, acc_bias_grad.rect, "[Linear:backward:bias_grad]");
    //print_tensor<2, float>(input_grad, acc_input.rect, "[Linear:backward:input_grad]");
  }
}

/*
  regions[0](I/O): input_grad
  regions[1](I): replicas
*/
__host__
void Linear::backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  float alpha = 1.0f;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  TensorAccessorW<float, 2> acc_input(
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
  int num_replica = acc_replica.rect.hi[2] - acc_replica.rect.lo[2] + 1;
  const float *replica_ptr = acc_replica.ptr;
  for (int i = 1; i < num_replica; i++) {
    checkCUDA(cublasSaxpy(m->handle.blas, acc_input.rect.volume(),
                          &alpha, replica_ptr, 1, acc_input.ptr, 1));
    replica_ptr += acc_input.rect.volume();
  }
}

void Linear::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
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
  if (!outputs[0].get_input_sub_tensor(pc, sub_input, OP_LINEAR))
    return false;
  int input_c = sub_input.adim[0];
  int input_n = sub_input.adim[1];
  int output_c = sub_output.adim[0];
  int output_n = sub_output.adim[1];
  LinearMeta* m = sim->linear_meta;
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

  // measure forward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    forward_kernel(m, input_ptr, output_ptr, kernel_ptr, bias_ptr,
        input_c, output_c, input_n);
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  forward_time = milliseconds / sim->repeat_times;

  // measure backward time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event));
    }
    backward_kernel(m, input_ptr, input_ptr, output_ptr, output_ptr,
        kernel_ptr, kernel_ptr, bias_ptr, input_c, output_c, input_n);
  }
  checkCUDA(cudaEventRecord(sim->end_event));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  backward_time = milliseconds / sim->repeat_times;

  printf("[Measure Linear] in(%d %d) out(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
         input_n, input_c, output_n, output_c, forward_time, backward_time);
  return true;
}


