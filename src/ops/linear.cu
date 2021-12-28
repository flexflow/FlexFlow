/* Copyright 2020 Stanford, NVIDIA, Facebook
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

#include "flexflow/ops/linear.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;

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
  assert(regions.size() == task->regions.size());
  assert(regions.size() == 2 || regions.size() == 3);
  const Linear* linear = (Linear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  //TensorAccessorR<float, 2> acc_input(
  //    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorW<float, 3> acc_kernel(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  // TensorAccessorR<float, 1> acc_bias(
  //     regions[3], task->regions[3], FID_DATA, ctx, runtime);
  //int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  printf("init linear (input): in_dim(%d) out_dim(%d) batch_size(%d)\n",
      in_dim, out_dim, batch_size);
  LinearMeta* m = new LinearMeta(handle, batch_size);
  m->activation = linear->activation;
  m->use_bias = linear->use_bias;
  m->profiling = linear->profiling;
  m->trainableInputs[0] = linear->trainableInputs[0];
  m->input_type = linear->inputs[0]->data_type;
  m->weight_type = linear->weights[0]->data_type;
  m->output_type = linear->outputs[0]->data_type;
  std::strcpy(m->op_name, linear->name);

  if (use_activation(m->activation)) {
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
                                          ff_to_cudnn_datatype(m->output_type),
                                          batch_size, out_dim, 1, 1));
  }
  return m;
}

/*static*/
void Linear::forward_kernel(const LinearMeta* m,
                            const void* input_ptr,
                            void* output_ptr,
                            const void* weight_ptr,
                            const void* bias_ptr,
                            int in_dim, int out_dim, int batch_size,
                            cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type);
#if CUDA_VERSION >= 11000 
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  checkCUDA(cublasGemmEx(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                         out_dim, batch_size, in_dim,
                         &alpha, weight_ptr, weight_type, in_dim,
                         input_ptr, input_type, in_dim, &beta,
                         output_ptr, output_type, out_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // use_bias = True 
  if (bias_ptr != NULL) { 
    checkCUDA(cublasGemmEx(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                           out_dim, batch_size, 1,
                           &alpha, bias_ptr, weight_type, 1,
                           m->one_ptr, CUDA_R_32F, 1, &alpha,
                           output_ptr, output_type, out_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (use_activation(m->activation)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr,
        &beta, m->outputTensor, output_ptr));
  } else if (m->activation == AC_MODE_GELU) {
    size_t elements = (size_t)out_dim * (size_t) batch_size;
    constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
    constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)
    gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS>>>(
        elements, B, C, (float*)output_ptr);
  } else if (m->activation == AC_MODE_NONE) {
    // Do nothing
  } else {
    assert(false && "Unsupported activation for Linear");
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
  //Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  assert(regions.size() == (3 + int(m->use_bias)));
  assert(task->regions.size() == (3 + int(m->use_bias)));
  
  TensorAccessorR<float, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 3> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_input.rect.volume() == in_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  const float* acc_bias_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorR<float, 3> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(acc_bias.rect.volume() == out_dim);
    acc_bias_ptr = acc_bias.ptr;
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Linear::forward_kernel(m, acc_input.ptr, acc_output.ptr,
      acc_kernel.ptr, acc_bias_ptr, in_dim, out_dim, batch_size, stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Linear] forward time = %.2lfms\n", m->op_name, elapsed);
    //print_tensor<float>(acc_input.ptr, acc_input.rect.volume(), "[Linear:forward:input]");
    //print_tensor<float>(acc_kernel.ptr, acc_kernel.rect.volume(), "[Linear:forward:kernel]");
    //print_tensor<float>(acc_bias_ptr, out_dim, "[Linear:forward:bias]");
    //print_tensor<float>(acc_output.ptr, acc_output.rect.volume(), "[Linear:forward:output]");
  }
}

/*static*/
void Linear::backward_kernel(const LinearMeta* m,
                             const void* input_ptr,
                             void* input_grad_ptr,
                             const void* output_ptr,
                             void* output_grad_ptr,
                             const void* kernel_ptr,
                             void* kernel_grad_ptr,
                             void* bias_grad_ptr,
                             int in_dim, int out_dim, int batch_size,
                             cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type);
#if CUDA_VERSION >= 11000 
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  int output_size = out_dim * batch_size;
  if (m->activation == AC_MODE_RELU) {
    relu_backward_kernel(m->output_type, output_grad_ptr, output_ptr, output_size, stream);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward_kernel(m->output_type, output_grad_ptr, output_ptr, output_size, stream);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(cublasGemmEx(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                         in_dim, out_dim, batch_size,
                         &alpha, input_ptr, input_type, in_dim,
                         output_grad_ptr, output_type, out_dim,
                         &alpha, kernel_grad_ptr, weight_type, in_dim,
                         compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  // use_bias = True
  if (bias_grad_ptr != NULL) {
    checkCUDA(cublasGemmEx(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                           1, out_dim, batch_size,
                           &alpha, m->one_ptr, CUDA_R_32F, 1,
                           output_grad_ptr, output_type, out_dim,
                           &alpha, bias_grad_ptr, weight_type, 1,
                           compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDA(cublasGemmEx(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                           in_dim, batch_size, out_dim,
                           &alpha, kernel_ptr, weight_type, in_dim,
                           output_grad_ptr, output_type, out_dim,
                           &alpha, input_grad_ptr, input_type, in_dim,
                           compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
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
  //Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  assert(regions.size() == (5 + int(m->trainableInputs[0]) + int(m->use_bias)));
  assert(task->regions.size() == (5 + int(m->trainableInputs[0]) + int(m->use_bias)));
  float* input_grad = NULL;
  size_t rid = 0;
  TensorAccessorR<float, NDIM> acc_input(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  if (m->trainableInputs[0]) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[rid].region.get_index_space());
    if (domain.get_dim() == NDIM+1) {
      assert(domain.get_volume() == acc_input.rect.volume());
      input_grad = helperGetTensorPointerWO<float>(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
    } else {
      TensorAccessorW<float, NDIM> acc_replica_grad(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      assert(acc_replica_grad.rect.volume() == acc_input.rect.volume());
      input_grad = acc_replica_grad.ptr;
    }
    rid++;
  }
  TensorAccessorR<float, NDIM> acc_output(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, NDIM> acc_output_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid++;
  TensorAccessorR<float, 3> acc_kernel(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, 3> acc_kernel_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid++;
  // make sure the sizes match
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_output_grad.rect.volume() == out_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  assert(acc_kernel_grad.rect.volume() == in_dim * out_dim);
  float* acc_bias_grad_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorW<float, 3> acc_bias_grad(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    rid++;
    assert(acc_bias_grad.rect.volume() == out_dim);
    acc_bias_grad_ptr = static_cast<float*>(acc_bias_grad.ptr);
  }
  assert(rid == regions.size());

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Linear::backward_kernel(m, acc_input.ptr, input_grad,
      acc_output.ptr, acc_output_grad.ptr,
      acc_kernel.ptr, acc_kernel_grad.ptr,
      acc_bias_grad_ptr, in_dim, out_dim, batch_size, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear backward time = %.2lfms\n", elapsed);
    //print_tensor<float>(acc_output_grad.ptr, acc_output_grad.rect.volume(), "[Linear:backward:output_grad]");
    //print_tensor<float>(acc_kernel_grad.ptr, acc_kernel_grad.rect.volume(), "[Linear:backward:kernel_grad]");
    //print_tensor<float>(acc_bias_grad_ptr, out_dim, "[Linear:backward:bias_grad]");
    //if (input_grad != nullptr)
    //  print_tensor<float>(input_grad, acc_input.rect.volume(), "[Linear:backward:input_grad]");
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

bool Linear::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, OP_LINEAR))
    return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_LINEAR))
    return false;
  int input_c = sub_input.dims[0].size;
  int input_n = sub_input.get_volume() / input_c;
  int output_c = sub_output.dims[0].size;
  int output_n = sub_output.get_volume() / output_c;
  LinearMeta* m = sim->linear_meta;
  m->activation = activation;
  m->input_type = inputs[0]->data_type;
  m->weight_type = this->data_type;
  m->output_type = outputs[0]->data_type;
  if (use_activation(m->activation)) {
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
  void* input_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  void* output_ptr = sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
  void* kernel_ptr = sim->allocate((size_t)output_c * input_c, this->data_type);
  void* bias_ptr = sim->allocate(output_c, this->data_type);
  assert(bias_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  bool out_of_memory = (input_ptr == NULL) || (output_ptr == NULL)
                       || (kernel_ptr == NULL) || (bias_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, kernel_ptr, bias_ptr,
        input_c, output_c, input_n, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    void* input_grad_ptr = NULL;
    if (trainableInputs[0]) {
      input_grad_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
    } else {
      input_grad_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
    }
    void* output_grad_ptr = sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
    void* kernel_grad_ptr = sim->allocate((size_t)output_c * input_c, this->data_type);
    void* bias_grad_ptr = sim->allocate(output_c, this->data_type);
    out_of_memory = (input_grad_ptr == NULL) || (output_grad_ptr == NULL)
                    || (kernel_grad_ptr == NULL) || (bias_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr,
          kernel_ptr, kernel_grad_ptr, bias_grad_ptr, input_c, output_c, input_n, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
           name, input_n, input_c, output_n, output_c,
           cost_metrics.forward_time, cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf)\n",
           name, input_n, input_c, output_n, output_c,
           cost_metrics.forward_time);
  }
  return true;
}

}; // namespace FlexFlow
