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

#include "internal/device.h"
#include "kernels/nccl.h"
#include "kernels/optimizer_kernels_gpu.h"

namespace FlexFlow {

__global__ void sgd_update(size_t count,
                           float lr,
                           float weight_decay,
                           float momentum,
                           bool nesterov,
                           float const *WGrad,
                           float *V,
                           float *W) {
  // Refernce https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
  CUDA_KERNEL_LOOP(i, count) {
    float gt = WGrad[i] + weight_decay * W[i];
    if (momentum > 0.0f) {
      V[i] = V[i] * momentum + gt;
      if (nesterov) {
        gt = gt + momentum * V[i];
      } else {
        gt = V[i];
      }
    }
    W[i] -= lr * gt;
  }
}

__host__ void gpu_sgd_ps_update_task(ffStream_t stream,
                                     float lr,
                                     float momentum,
                                     bool nesterov,
                                     float weight_decay,
                                     float const *weight_grad_ptr,
                                     size_t size,
                                     int num_replicas,
                                     float *weight_ptr,
                                     float *sgd_v_ptr) {
  // Step 1: Gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    float const *src = weight_grad_ptr + i * size;
    apply_add_with_scale<float>
        <<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)weight_grad_ptr, src, size, 1.0f);
  }

  //  Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(size,
                                                                lr,
                                                                weight_decay,
                                                                momentum,
                                                                nesterov,
                                                                weight_grad_ptr,
                                                                sgd_v_ptr,
                                                                weight_ptr);
}

__host__ void gpu_sgd_nccl_update_task(ffStream_t stream,
                                       float lr,
                                       float momentum,
                                       bool nesterov,
                                       float weight_decay,
                                       PerDeviceFFHandle const &handle,
                                       float const *w_grad_ptr,
                                       size_t size,
                                       float *w_ptr,
                                       float *v_ptr) {
  // Step 1: Use NCCL to sync gradients
  ncclComm_t comm = handle.ncclComm;
  checkNCCL(ncclAllReduce(
      w_grad_ptr, (float *)w_grad_ptr, size, ncclFloat, ncclSum, comm, stream));

  //  Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size, lr, weight_decay, momentum, nesterov, w_grad_ptr, v_ptr, w_ptr);
}

// ==================================================================
//                        Adam Optimizer
// ==================================================================
__global__ void
    add_kernel(int count, float scale, float const *src, float *dst) {
  CUDA_KERNEL_LOOP(i, count) {
    dst[i] += src[i] * scale;
  }
}

__global__ void scale_kernel(int count, float a, float b, float *ptr) {
  CUDA_KERNEL_LOOP(i, count) {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__ void adam_update(int count,
                            float alpha_t,
                            float beta1,
                            float beta2,
                            float weight_decay,
                            float epsilon,
                            float const *WGrad,
                            float *M,
                            float *V,
                            float *W) {
  // Reference for weight decay
  // https://www.fast.ai/2018/07/02/adam-weight-decay/
  CUDA_KERNEL_LOOP(i, count) {
    // W[i] -= weight_decay * alpha_t * W[i];
    // float gt = WGrad[i];
    float gt = WGrad[i] + weight_decay * W[i];
    float mt = beta1 * M[i] + (1 - beta1) * gt;
    float vt = beta2 * V[i] + (1 - beta2) * gt * gt;
    M[i] = mt;
    V[i] = vt;
    W[i] -= alpha_t * mt / (sqrt(vt) + epsilon);
  }
}

__host__ void gpu_adam_ps_update_task(ffStream_t stream,
                                      float alpha_t,
                                      float beta1,
                                      float beta2,
                                      float weight_decay,
                                      float epsilon,
                                      float const *w_grad_ptr,
                                      size_t size,
                                      int num_replicas,
                                      float *w_ptr,
                                      float *v_ptr,
                                      float *m_ptr) {
  // Step 1: Gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    float const *src = w_grad_ptr + i * size;
    add_kernel<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
        (float *)w_grad_ptr, src, size);
  }

  //  Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(size,
                                                                 alpha_t,
                                                                 beta1,
                                                                 beta2,
                                                                 weight_decay,
                                                                 epsilon,
                                                                 w_grad_ptr,
                                                                 m_ptr,
                                                                 v_ptr,
                                                                 w_ptr);
}

__host__ void gpu_adam_nccl_update_task(ffStream_t stream,
                                        float alpha_t,
                                        float beta1,
                                        float beta2,
                                        float weight_decay,
                                        float epsilon,
                                        PerDeviceFFHandle const &handle,
                                        float const *w_grad_ptr,
                                        size_t size,
                                        float *w_ptr,
                                        float *v_ptr,
                                        float *m_ptr) {
  // Step 1: Use NCCL to sync gradients
  checkNCCL(ncclAllReduce(w_grad_ptr,
                          (float *)w_grad_ptr,
                          size,
                          ncclFloat,
                          ncclSum,
                          handle.ncclComm,
                          stream));

  //  Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(size,
                                                                 alpha_t,
                                                                 beta1,
                                                                 beta2,
                                                                 weight_decay,
                                                                 epsilon,
                                                                 w_grad_ptr,
                                                                 m_ptr,
                                                                 v_ptr,
                                                                 w_ptr);
}

} // namespace FlexFlow
