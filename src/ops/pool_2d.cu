/* Copyright 2018 Stanford
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

#include "flexflow/ops/pool_2d.h"
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
  regions[0]: input
  regions[1]: output
*/
OpMeta* Pool2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Pool2D* pool = (Pool2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  Pool2DMeta* m = new Pool2DMeta(handle);
  m->profiling = pool->profiling;
  std::strcpy(m->op_name, pool->name);
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

  int input_w = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_h = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int input_c = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int input_n = acc_input.rect.hi[3] - acc_input.rect.lo[3] + 1;
  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int output_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  int output_n = acc_output.rect.hi[3] - acc_output.rect.lo[3] + 1;

  printf("init pool (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n, input_c, input_h, input_w);
  printf("init pool (output): n(%d) c(%d) h(%d) w(%d)\n",
         output_n, output_c, output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
  int pad_h = ((output_h - 1) * pool->stride_h + pool->kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * pool->stride_w + pool->kernel_w - input_w + 1) / 2;
  if (pad_h != pool->padding_h)
    printf("Warning: changing pool_padding_h to satisfy output_h size\n");
  if (pad_w != pool->padding_w)
    printf("Warning: changing pool_padding_w to satisfy output_w size\n");

  cudnnPoolingMode_t mode;
  if (pool->pool_type == POOL_MAX)
    mode = CUDNN_POOLING_MAX;
  else {
    assert(pool->pool_type == POOL_AVG);
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         pool->kernel_h,
                                         pool->kernel_w,
                                         pad_h,//pool->padding_h,
                                         pad_w,//pool->padding_w,
                                         pool->stride_h,
                                         pool->stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(m->poolDesc,
                                               m->inputTensor,
                                               &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  return m;
}

/*static*/
void Pool2D::forward_kernel(const Pool2DMeta* m,
                            const float* input_ptr,
                            float* output_ptr,
                            cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(m->handle.dnn, m->poolDesc,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->outputTensor, output_ptr));
}

/*
  regions[0](I): input
  regions[1](O): output
*/
void Pool2D::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Pool2D* pool = (Pool2D*) task->args;
  const Pool2DMeta* m = *((Pool2DMeta**) task->local_args);
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Output::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  forward_kernel(m, acc_input.ptr, acc_output.ptr, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<4, float>(acc_input.ptr, acc_input.rect, "[Pool2D:forward:input]");
    //print_tensor<4, float>(acc_output.ptr, acc_output.rect, "[Pool2D:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Pool2D] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

/*static*/
void Pool2D::backward_kernel(const Pool2DMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             const float* output_grad_ptr,
                             cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  checkCUDNN(cudnnPoolingBackward(m->handle.dnn, m->poolDesc,
                                  &alpha, m->outputTensor, output_ptr,
                                  m->outputTensor, output_grad_ptr,
                                  m->inputTensor, input_ptr,
                                  &alpha, m->inputTensor, input_grad_ptr));
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
void Pool2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  //const Pool2D* pool = (Pool2D*) task->args;
  const Pool2DMeta* m = *((Pool2DMeta**) task->local_args);
  TensorAccessorR<float, Input::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Input::NUMDIM> acc_input_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, Output::NUMDIM> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, Output::NUMDIM> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  backward_kernel(m, acc_input.ptr, acc_input_grad.ptr, acc_output.ptr, acc_output_grad.ptr, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Pool2D backward time = %.2fms\n", elapsed);
  }
}

Pool2DMeta::Pool2DMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
}

bool Pool2D::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_input;
  if(!outputs[0]->get_output_sub_tensor(pc, sub_output, OP_POOL2D))
    return false;
  if(!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_POOL2D))
    return false;
  int input_w = sub_input.dims[0].size;
  int input_h = sub_input.dims[1].size;
  int input_c = sub_input.dims[2].size;
  int input_n = sub_input.dims[3].size;
  int output_w = sub_output.dims[0].size;
  int output_h = sub_output.dims[1].size;
  int output_c = sub_output.dims[2].size;
  int output_n = sub_output.dims[3].size;
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;
  Pool2DMeta* m = sim->pool2d_meta;
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));
  cudnnPoolingMode_t mode;
  if (pool_type == POOL_MAX)
    mode = CUDNN_POOLING_MAX;
  else {
    assert(pool_type == POOL_AVG);
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         kernel_h,
                                         kernel_w,
                                         pad_h,//pool->padding_h,
                                         pad_w,//pool->padding_w,
                                         stride_h,
                                         stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(m->poolDesc,
                                               m->inputTensor,
                                               &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    float *output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    backward = [&] {
      backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug(
        "[Measure Pool2D] name(%s) input(%d %d %d %d) output(%d %d %d %d) stride(%d %d) padding(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        input_n, input_c, input_h, input_w,
        output_n, output_c, output_h, output_w,
        stride_h, stride_w,
        padding_h, padding_w,
        cost_metrics.forward_time, cost_metrics.backward_time);
  } else {
    log_measure.debug(
        "[Measure Pool2D] name(%s) input(%d %d %d %d) output(%d %d %d %d) stride(%d %d) padding(%d %d) forward_time(%.4lf)\n",
        name,
        input_n, input_c, input_h, input_w,
        output_n, output_c, output_h, output_w,
        stride_h, stride_w,
        padding_h, padding_w,
        cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow 
