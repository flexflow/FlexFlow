/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/ops/kernels/pool_2d_kernels.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {

Pool2DMeta::Pool2DMeta(FFHandler handler) : OpMeta(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreatePoolingDescriptor(&poolDesc));
}

namespace Kernels {
namespace Pool2D {

void init_kernel(Pool2DMeta *m,
                 int input_w,
                 int input_h,
                 int input_c,
                 int input_n,
                 int output_w,
                 int output_h,
                 int output_c,
                 int output_n,
                 int pad_h,
                 int pad_w,
                 int kernel_h,
                 int kernel_w,
                 int stride_h,
                 int stride_w,
                 PoolType pool_type) {
  checkCUDNN(miopenSet4dTensorDescriptor(
      m->inputTensor, miopenFloat, input_n, input_c, input_h, input_w));

  miopenPoolingMode_t mode;
  if (pool_type == POOL_MAX) {
    mode = miopenPoolingMax;
  } else {
    assert(pool_type == POOL_AVG);
    mode = miopenPoolingAverage;
  }
  checkCUDNN(miopenSet2dPoolingDescriptor(
      m->poolDesc, mode, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
  int n, c, h, w;
  checkCUDNN(miopenGetPoolingForwardOutputDim(
      m->poolDesc, m->inputTensor, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(
      miopenSet4dTensorDescriptor(m->outputTensor, miopenFloat, n, c, h, w));
}

void forward_kernel_wrapper(Pool2DMeta const *m,
                            void const *input_ptr,
                            void *output_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::forward_kernel(m, input_ptr, output_ptr, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    // print_tensor<4, float>(acc_input.ptr, acc_input.rect,
    // "[Pool2D:forward:input]"); print_tensor<4, float>(acc_output.ptr,
    // acc_output.rect, "[Pool2D:forward:output]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("%s [Pool2D] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(Pool2DMeta const *m,
                             void const *input_ptr,
                             void *input_grad_ptr,
                             void const *output_ptr,
                             void const *output_grad_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::backward_kernel(
      m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("Pool2D backward time = %.2fms\n", elapsed);
  }
}

namespace Internal {

void forward_kernel(Pool2DMeta const *m,
                    void const *input_ptr,
                    void *output_ptr,
                    hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenPoolingForward(m->handle.dnn,
                                  m->poolDesc,
                                  &alpha,
                                  m->inputTensor,
                                  input_ptr,
                                  &beta,
                                  m->outputTensor,
                                  output_ptr,
                                  true,
                                  m->handle.workSpace,
                                  m->handle.workSpaceSize));
}

void backward_kernel(Pool2DMeta const *m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void const *output_grad_ptr,
                     hipStream_t stream) {
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  float beta = 0.0f;
  checkCUDNN(miopenPoolingBackward(m->handle.dnn,
                                   m->poolDesc,
                                   &alpha,
                                   m->outputTensor,
                                   output_ptr,
                                   m->outputTensor,
                                   output_grad_ptr,
                                   m->inputTensor,
                                   input_ptr,
                                   &beta,
                                   m->inputTensor,
                                   input_grad_ptr,
                                   m->handle.workSpace));
}

} // namespace Internal
} // namespace Pool2D
} // namespace Kernels
} // namespace FlexFlow
