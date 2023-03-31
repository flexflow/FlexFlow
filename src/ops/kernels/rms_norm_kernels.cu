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

#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/utils/cuda_helper.h"
#include <cublas_v2.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

RMSNormMeta::RMSNormMeta(FFHandler handler,
                         RMSNorm const *rms,
                         coord_t _in_dim,
                         coord_t _num_dims)
    : OpMeta(handler, rms) {
  eps = rms->eps;
  in_dim = _in_dim;
  num_dims = _num_dims;
  num_elements = in_dim * num_dims;

  checkCUDA(cudaMalloc(&mean_ptr, in_dim * num_dims * sizeof(float)));

  // set descriptor for reduce kenrnel
  checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_dims, in_dim, 1));

  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        num_dims,
                                        in_dim,
                                        1));

  checkCUDNN(cudnnSetReduceTensorDescriptor(reduceDesc,
                                            CUDNN_REDUCE_TENSOR_AVG,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_PROPAGATE_NAN,
                                            CUDNN_REDUCE_TENSOR_NO_INDICES,
                                            CUDNN_32BIT_INDICES));
  // checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, input_domain));
  // Domain output_domain = input_domain;
  // for (size_t i = 0; i < rd->num_axes; i++) {
  //   assert(input_domain.dim > rd->axes[i]);
  //   output_domain.rect_data[rd->axes[i] + output_domain.dim] =
  //       output_domain.rect_data[rd->axes[i]];
  // }
  // checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor,
  // output_domain));
}

namespace Kernels {
namespace RMSNorm {

void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Internal::forward_kernel(m,
                           input.get_float_ptr(),
                           weight.get_float_ptr(),
                           output.get_float_ptr(),
                           input.domain.get_volume(),
                           stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[RMSNorm] forward time (CF) = %.2fms\n", elapsed);
    print_tensor<float>(input.get_float_ptr(), 32, "[RMSNorm:forward:input]");
    print_tensor<float>(output.get_float_ptr(), 32, "[RMSNorm:forward:output]");
  }
}

namespace Internal {
/*static*/
void norm_kernel(RMSNormMeta const *m,
                 float const *input_ptr,
                 float *output_ptr,
                 cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));

  float alpha = 2.0f;
  float beta = 0.0f;

  checkCUDA(cublasSgeam(m->handle.blas,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        m->num_dims,
                        m->in_dim,
                        &alpha,
                        input_ptr,
                        m->in_dim,
                        &beta,
                        input_ptr,
                        m->in_dim,
                        output_ptr,
                        m->num_dims));

  cublasSaxpy(handle, m->num_elements, &m->eps, output_ptr, 1, output_ptr, 1);
}

void forward_kernel(RMSNormMeta const *m,
                    float const *input_ptr,
                    float const *weight_ptr,
                    float *output_ptr,
                    coord_t dim_size,
                    cudaStream_t stream) {

  // impl from
  // https://github.com/facebookresearch/llama/blob/main/llama/model.py#:~:text=class%20RMSNorm(torch,*%20self.weight
  // pow
  norm_kernel(m, input_ptr, m->mean_ptr, stream);

  // reduce

  // add eps

  // multiply with x

  // apply weights

  return;
}
} // namespace Internal
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow