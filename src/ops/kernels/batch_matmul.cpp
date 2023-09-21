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

#include "flexflow/ops/kernels/batch_matmul_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

BatchMatmulMeta::BatchMatmulMeta(FFHandler handler) : OpMeta(handler) {}

namespace Kernels {
namespace BatchMatmul {

void forward_kernel_wrapper(BatchMatmulMeta const *meta,
                            float *o_ptr,
                            float const *a_ptr,
                            float const *b_ptr,
                            float const *c_ptr,
                            int m,
                            int n,
                            int k,
                            int batch,
                            int a_seq_length_dim,
                            int b_seq_length_dim,
                            int seq_length) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (meta->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::forward_kernel(meta,
                           o_ptr,
                           a_ptr,
                           b_ptr,
                           c_ptr,
                           m,
                           n,
                           k,
                           batch,
                           stream,
                           a_seq_length_dim,
                           b_seq_length_dim,
                           seq_length);
  if (meta->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("BatchMatmul forward time = %.2lfms\n", elapsed);
  }
}

void backward_kernel_wrapper(BatchMatmulMeta const *meta,
                             float const *o_ptr,
                             float const *o_grad_ptr,
                             float const *a_ptr,
                             float *a_grad_ptr,
                             float const *b_ptr,
                             float *b_grad_ptr,
                             float *c_grad_ptr,
                             int m,
                             int n,
                             int k,
                             int batch) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (meta->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }
  Internal::backward_kernel(meta,
                            o_ptr,
                            o_grad_ptr,
                            a_ptr,
                            a_grad_ptr,
                            b_ptr,
                            b_grad_ptr,
                            c_grad_ptr,
                            m,
                            n,
                            k,
                            batch,
                            stream);
  if (meta->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("BatchMatmul backward time = %.2lfms\n", elapsed);
  }
}

namespace Internal {

/*
A: (batch, n, k)
B: (batch, k, m)
O: (batch, n, m)
O = A * B
*/
void forward_kernel(BatchMatmulMeta const *meta,
                    float *o_ptr,
                    float const *a_ptr,
                    float const *b_ptr,
                    float const *c_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    hipStream_t stream,
                    int a_seq_length_dim,
                    int b_seq_length_dim,
                    int seq_length) {
  checkCUDA(hipblasSetStream(meta->handle.blas, stream));
  checkCUDNN(miopenSetStream(meta->handle.dnn, stream));

  // int a_stride = n * k;
  // int b_stride = m * k;
  // int o_stride = n * m;
  int lda = k;
  int ldb = m;
  int ldo = m;
  long long int strideA = (long long int)n * k;
  long long int strideB = (long long int)k * m;
  long long int strideO = (long long int)n * m;
  if ((a_seq_length_dim == 0) && (seq_length >= 0)) {
    assert(seq_length <= k);
    k = seq_length;
    assert(b_seq_length_dim == 1);
  } else if ((a_seq_length_dim == 1) && (seq_length >= 0)) {
    assert(seq_length <= n);
    n = seq_length;
  } else {
    // currently only support a_seq_length_dim = 0 or 1
    assert((a_seq_length_dim < 0) || (seq_length < 0));
  }
  if ((b_seq_length_dim == 0) && (seq_length >= 0)) {
    assert(seq_length <= m);
    m = seq_length;
  } else if ((b_seq_length_dim == 1) && (seq_length >= 0)) {
    assert(a_seq_length_dim == 0);
    assert(k == seq_length);
  } else {
    // currently only support a_seq_length_dim = 0 or 1
    assert((b_seq_length_dim < 0) || (seq_length < 0));
  }

  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(hipblasSgemmStridedBatched(meta->handle.blas,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       b_ptr,
                                       ldb,
                                       strideB,
                                       a_ptr,
                                       lda,
                                       strideA,
                                       &beta,
                                       o_ptr,
                                       ldo,
                                       strideO,
                                       batch));
  // current assume c is null
  assert(c_ptr == NULL);
}

/*
A, AGrad: (batch, n, k)
B, BGrad: (batch, k, m)
O, OGrad: (batch, n, m)
AGrad = OGrad * B^T
BGrad = A^T * OGrad
*/
void backward_kernel(BatchMatmulMeta const *meta,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     float *c_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch,
                     hipStream_t stream) {
  checkCUDA(hipblasSetStream(meta->handle.blas, stream));
  checkCUDNN(miopenSetStream(meta->handle.dnn, stream));

  int a_stride = n * k;
  int b_stride = m * k;
  int o_stride = n * m;
  float alpha = 1.0f;
  checkCUDA(hipblasSgemmStridedBatched(meta->handle.blas,
                                       HIPBLAS_OP_T,
                                       HIPBLAS_OP_N,
                                       k,
                                       n,
                                       m,
                                       &alpha,
                                       b_ptr,
                                       m,
                                       b_stride,
                                       o_grad_ptr,
                                       m,
                                       o_stride,
                                       &alpha,
                                       a_grad_ptr,
                                       k,
                                       a_stride,
                                       batch));
  checkCUDA(hipblasSgemmStridedBatched(meta->handle.blas,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_T,
                                       m,
                                       k,
                                       n,
                                       &alpha,
                                       o_grad_ptr,
                                       m,
                                       o_stride,
                                       a_ptr,
                                       k,
                                       a_stride,
                                       &alpha,
                                       b_grad_ptr,
                                       m,
                                       b_stride,
                                       batch));
  assert(c_grad_ptr == NULL);
}

} // namespace Internal
} // namespace BatchMatmul
} // namespace Kernels
} // namespace FlexFlow
