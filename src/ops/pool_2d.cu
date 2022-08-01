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

/*static*/
void Pool2D::init_kernel(const Pool2D *pool,
                         Pool2DMeta *m,
                         int input_w,
                         int input_h,
                         int input_c,
                         int input_n,
                         int output_w,
                         int output_h,
                         int output_c,
                         int output_n,
                         int pad_h,
                         int pad_w) {
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

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
                                         pad_h, // pool->padding_h,
                                         pad_w, // pool->padding_w,
                                         pool->stride_h,
                                         pool->stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(
      m->poolDesc, m->inputTensor, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(
      m->outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
}

/*static*/
void Pool2D::forward_kernel(const Pool2DMeta *m,
                            const float *input_ptr,
                            float *output_ptr,
                            cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(m->handle.dnn,
                                 m->poolDesc,
                                 &alpha,
                                 m->inputTensor,
                                 input_ptr,
                                 &beta,
                                 m->outputTensor,
                                 output_ptr));
}

/*static*/
void Pool2D::forward_kernel_wrapper(const Pool2DMeta *m,
                                    const float *input_ptr,
                                    float *output_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Pool2D::forward_kernel(m, input_ptr, output_ptr, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<4, float>(acc_input.ptr, acc_input.rect,
    // "[Pool2D:forward:input]"); print_tensor<4, float>(acc_output.ptr,
    // acc_output.rect, "[Pool2D:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Pool2D] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

/*static*/
void Pool2D::backward_kernel(const Pool2DMeta *m,
                             const float *input_ptr,
                             float *input_grad_ptr,
                             const float *output_ptr,
                             const float *output_grad_ptr,
                             cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  checkCUDNN(cudnnPoolingBackward(m->handle.dnn,
                                  m->poolDesc,
                                  &alpha,
                                  m->outputTensor,
                                  output_ptr,
                                  m->outputTensor,
                                  output_grad_ptr,
                                  m->inputTensor,
                                  input_ptr,
                                  &alpha,
                                  m->inputTensor,
                                  input_grad_ptr));
}

/*static*/
void Pool2D::backward_kernel_wrapper(const Pool2DMeta *m,
                                     const float *input_ptr,
                                     float *input_grad_ptr,
                                     const float *output_ptr,
                                     const float *output_grad_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Pool2D::backward_kernel(
      m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr, stream);
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

Pool2DMeta::Pool2DMeta(FFHandler handler) : OpMeta(handler) {
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
}

}; // namespace FlexFlow