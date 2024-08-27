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

#include "device.h"
#include "kernels/optimizer_kernels.h"
#include "kernels/nccl.h"

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

void sgd_ps_update_task_gpu(cudaStream_t stream,
                            float lr,
                            float momentum,
                            bool nesterov,
                            float weight_decay,
                            float const *weight_grad_ptr,
                            size_t size,
                            int num_replicas,
                            float *weight_ptr,
                            float *sgd_v_ptr) {
  checkCUDA(get_legion_stream(&stream));
  // Step 1: Gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    float const *src = weight_grad_ptr + i * size;
    apply_add_with_scale<float>
        <<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)weight_grad_ptr, src, size, 1.0f);
  }
  // checkCUDA(cudaDeviceSynchronize());
  //  Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      lr,
      weight_decay,
      momentum,
      nesterov,
      weight_grad_ptr,
      sgd_v_ptr,
      weight_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}

#ifdef FF_USE_NCCL
void sgd_nccl_update_task_gpu(cudaStream_t stream,
                              float lr,
                              float momentum,
                              bool nesterov,
                              float weight_decay,
                              PerDeviceFFHandle const & handle,
                              float const *weight_grad_ptr,
                              size_t size,
                              float *weight_ptr,
                              float *sgd_v_ptr) {
  // Use NCCL to sync gradients
  // fprintf(stderr, "weight(%p) Before ncclAllReduce...\n", w_grad_ptr);
  checkCUDA(get_legion_stream(&stream));
  checkNCCL(ncclAllReduce(weight_grad_ptr,
                          (float *)weight_grad_ptr,
                          size,
                          ncclDataType_t::ncclFloat,
                          ncclRedOp_t::ncclSum,
                          handle.ncclComm,
                          stream));
  // fprintf(stderr, "weight(%p) After ncclAllReduce...\n", w_grad_ptr);
  // print_tensor<float>((float*)w_grad_ptr, 16, "[After ncclAllReduce]");

  // Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      lr,
      weight_decay,
      momentum,
      nesterov,
      weight_grad_ptr,
      sgd_v_ptr,
      weight_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}
#endif

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

void adam_ps_update_task_gpu(cudaStream_t stream,
                             float alpha_t,
                             float beta1,
                             float beta2,
                             float weight_decay,
                             float epsilon,
                             size_t size,
                             int num_replicas,
                             float const *weight_grad_ptr,
                             float *adam_m_ptr,
                             float *adam_v_ptr,
                             float *weight_ptr) {
  checkCUDA(get_legion_stream(&stream));
  // Step 1: Gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    float const *src = weight_grad_ptr + i * size;
    add_kernel<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
        size, 1.0f, src, (float *)weight_grad_ptr);
  }
  // checkCUDA(cudaDeviceSynchronize());
  // fprintf(stderr, "alpha = %.8lf alpha_t = %.8lf decay = %.8lf\n",
  //         op->alpha, op->alpha_t, op->weight_decay);
  //  Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      alpha_t,
      beta1,
      beta2,
      weight_decay,
      epsilon,
      weight_grad_ptr,
      adam_m_ptr,
      adam_v_ptr,
      weight_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}

#ifdef FF_USE_NCCL
void adam_nccl_update_task_gpu(cudaStream_t stream,
                               float alpha_t,
                               float beta1,
                               float beta2,
                               float weight_decay,
                               float epsilon,
                               size_t size,
                               PerDeviceFFHandle const & handle,
                               float const *weight_grad_ptr,
                               float *adam_m_ptr,
                               float *adam_v_ptr,
                               float *weight_ptr) {
  // Use NCCL to sync gradients
  checkCUDA(get_legion_stream(&stream));
  checkNCCL(ncclAllReduce(weight_grad_ptr,
                          (float *)weight_grad_ptr,
                          size,
                          ncclDataType_t::ncclFloat,
                          ncclRedOp_t::ncclSum,
                          handle.ncclComm,
                          stream));
  // fprintf(stderr, "alpha = %.8lf alpha_t = %.8lf decay = %.8lf\n",
  //         op->alpha, op->alpha_t, op->weight_decay);
  //  Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      alpha_t,
      beta1,
      beta2,
      weight_decay,
      epsilon,
      weight_grad_ptr,
      adam_m_ptr,
      adam_v_ptr,
      weight_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}
#endif

} // namespace FlexFlow
