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

#include "flexflow/accessor.h"
#include "flexflow/model.h"
#include "flexflow/optimizer.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

LegionRuntime::Logger::Category log_optimizer("optimizer");

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

__host__ void SGDOptimizer::ps_update_task_gpu(SGDOptimizer const *op,
                                               float const *w_grad_ptr,
                                               size_t size,
                                               int num_replicas,
                                               float *w_ptr,
                                               float *v_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Step 1: Gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    float const *src = w_grad_ptr + i * size;
    apply_add_with_scale<float>
        <<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
            (float *)w_grad_ptr, src, size, 1.0f);
  }
  // checkCUDA(cudaDeviceSynchronize());
  //  Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      op->lr,
      op->weight_decay,
      op->momentum,
      op->nesterov,
      w_grad_ptr,
      v_ptr,
      w_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}

#ifdef FF_USE_NCCL
__host__ void SGDOptimizer::nccl_update_task_gpu(SGDOptimizer const *op,
                                                 OpMeta const *meta,
                                                 float const *w_grad_ptr,
                                                 size_t size,
                                                 float *w_ptr,
                                                 float *v_ptr) {
  // Use NCCL to sync gradients
  // fprintf(stderr, "weight(%p) Before ncclAllReduce...\n", w_grad_ptr);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkNCCL(ncclAllReduce(w_grad_ptr,
                          (float *)w_grad_ptr,
                          size,
                          ncclFloat,
                          ncclSum,
                          meta->handle.ncclComm,
                          stream));
  // fprintf(stderr, "weight(%p) After ncclAllReduce...\n", w_grad_ptr);
  // print_tensor<float>((float*)w_grad_ptr, 16, "[After ncclAllReduce]");

  // Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      op->lr,
      op->weight_decay,
      op->momentum,
      op->nesterov,
      w_grad_ptr,
      v_ptr,
      w_ptr);
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

__host__ void AdamOptimizer::ps_update_task_gpu(AdamOptimizer const *op,
                                                float const *w_grad_ptr,
                                                size_t size,
                                                int num_replicas,
                                                float *w_ptr,
                                                float *v_ptr,
                                                float *m_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Step 1: Gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    float const *src = w_grad_ptr + i * size;
    add_kernel<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
        size, 1.0f, src, (float *)w_grad_ptr);
  }
  // checkCUDA(cudaDeviceSynchronize());
  // fprintf(stderr, "alpha = %.8lf alpha_t = %.8lf decay = %.8lf\n",
  //         op->alpha, op->alpha_t, op->weight_decay);
  //  Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      op->alpha_t,
      op->beta1,
      op->beta2,
      op->weight_decay,
      op->epsilon,
      w_grad_ptr,
      m_ptr,
      v_ptr,
      w_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}

#ifdef FF_USE_NCCL
__host__ void AdamOptimizer::nccl_update_task_gpu(AdamOptimizer const *op,
                                                  OpMeta const *meta,
                                                  float const *w_grad_ptr,
                                                  size_t size,
                                                  float *w_ptr,
                                                  float *v_ptr,
                                                  float *m_ptr) {
  // Use NCCL to sync gradients
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkNCCL(ncclAllReduce(w_grad_ptr,
                          (float *)w_grad_ptr,
                          size,
                          ncclFloat,
                          ncclSum,
                          meta->handle.ncclComm,
                          stream));
  // fprintf(stderr, "alpha = %.8lf alpha_t = %.8lf decay = %.8lf\n",
  //         op->alpha, op->alpha_t, op->weight_decay);
  //  Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      size,
      op->alpha_t,
      op->beta1,
      op->beta2,
      op->weight_decay,
      op->epsilon,
      w_grad_ptr,
      m_ptr,
      v_ptr,
      w_ptr);
  // checkCUDA(cudaDeviceSynchronize());
}

__host__ void AdamOptimizer::nccl_unified_update_task_gpu(
    AdamOptimizer const *op,
    OpMeta const *meta,
    GenericTensorAccessorR *accWGrads,
    size_t *size,
    GenericTensorAccessorW *accWs,
    GenericTensorAccessorW *accVs,
    GenericTensorAccessorW *accMs) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // assert(op->reservedWorkSpaceSize < meta->handle.workSpaceSize);

  cudaEvent_t t_start, t_start1, t_start2, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_start1);
  cudaEventCreate(&t_start2);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, stream);
  cudaEventRecord(t_start1, stream);
  cudaEventRecord(t_start2, stream);

  void *allocate_ptr;
  //  = meta->handle.workSpace;
  checkCUDA(
      cudaMalloc(&allocate_ptr,meta->handle.workSpaceSize));
  
  void *workSpace_ptr = allocate_ptr;

  for (int i = 0; i < op->parameters_num; i++) {
    cudaMemcpyAsync(workSpace_ptr,
                    accWGrads[i].get_float_ptr(),
                    size[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream);
    workSpace_ptr =
        static_cast<char *>(workSpace_ptr) + size[i] * sizeof(float);
  }

  cudaEventRecord(t_end, stream);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start1, t_end));
  cudaEventDestroy(t_start1);
  printf("[optimizer] data copy time = %.2lfms\n", elapsed);

  // do allreduce once
  checkNCCL(ncclAllReduce(meta->handle.workSpace,
                          (float *)meta->handle.workSpace,
                          meta->handle.workSpaceSize,
                          ncclFloat,
                          ncclSum,
                          meta->handle.ncclComm,
                          stream));
  cudaEventRecord(t_end, stream);
  checkCUDA(cudaEventSynchronize(t_end));
  elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start2, t_end));
  cudaEventDestroy(t_start2);
  printf("[optimizer] allreduce time = %.2lfms\n", elapsed);

  // workSpace_ptr = static_cast<char *>(meta->handle.workSpace);
  workSpace_ptr = static_cast<char *>(allocate_ptr);
  float alpha_t = op->alpha_t;
  float beta1_t = op->beta1_t;
  float beta2_t = op->beta2_t;
  for (int i = 0; i < op->parameters_num; i++) {
    // update
    // printf("update %d\n", i);
    adam_update<<<GET_BLOCKS(size[i]), CUDA_NUM_THREADS, 0, stream>>>(
        size[i],
        alpha_t,
        op->beta1,
        op->beta2,
        op->weight_decay,
        op->epsilon,
        static_cast<float *>(workSpace_ptr),
        accMs[i].get_float_ptr(),
        accVs[i].get_float_ptr(),
        accWs[i].get_float_ptr());
    workSpace_ptr =
        static_cast<char *>(workSpace_ptr) + size[i] * sizeof(float);

    // update
    beta1_t *= op->beta1;
    beta2_t *= op->beta2;
    alpha_t = op->alpha * sqrt(1 - beta2_t) / (1 - beta1_t);
  }
  cudaEventRecord(t_end, stream);
  checkCUDA(cudaEventSynchronize(t_end));
  elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  checkCUDA(cudaFree(allocate_ptr));
  printf("[optimizer] total time = %.2lfms\n", elapsed);
}
#endif

}; // namespace FlexFlow
