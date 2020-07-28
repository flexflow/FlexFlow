/* Copyright 2019 Stanford
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

#include "optimizer.h"
#include "accessor.h"
#include "model.h"
#include "cuda_helper.h"

LegionRuntime::Logger::Category log_optimizer("optimizer");

__global__
void sgd_update(int count, float lr, float weight_decay,
                float momentum, bool nesterov,
                const float* WGrad, float* V, float* W)
{
  // Refernce https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
  CUDA_KERNEL_LOOP(i, count)
  {
    float gt = WGrad[i] + weight_decay * W[i];
    if (momentum > 0.0f) {
      V[i] = V[i] * momentum + gt;
      if (nesterov)
        gt = gt + momentum * V[i];
      else
        gt = V[i];
    }
    W[i] -= lr * gt;
  }
}

__host__
void SGDOptimizer::update_task(const Task* task,
                               const std::vector<PhysicalRegion>& regions,
                               Context ctx, Runtime* runtime)
{
  const SGDOptimizer* op = (SGDOptimizer*) task->args;
  if (op->momentum > 0.0f) {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
  }
  Domain domain = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  const float *w_grad_ptr = NULL;
  float *w_ptr = NULL, *v_ptr = NULL;
  size_t size = 0, num_replicas = 0;
  switch(domain.get_dim()) {
    case 1:
    {
      TensorAccessorR<float, 1> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 1> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      for (int i = 0; i < domain.get_dim()-1; i++) {
        assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);
        assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);
      }
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      if (op->momentum > 0.0f) {
        TensorAccessorW<float, 1> accV(
            regions[2], task->regions[2], FID_DATA, ctx, runtime,
            true/*readOutput*/);
        assert(accW.rect == accV.rect);
        v_ptr = accV.ptr;
      }
      break;
    }
    case 2:
    {
      TensorAccessorR<float, 2> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 2> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      for (int i = 0; i < domain.get_dim()-1; i++) {
        assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);
        assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);
      }
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      if (op->momentum > 0.0f) {
        TensorAccessorW<float, 2> accV(
            regions[2], task->regions[2], FID_DATA, ctx, runtime,
            true/*readOutput*/);
        assert(accW.rect == accV.rect);
        v_ptr = accV.ptr;
      }
      break;
    }
    case 3:
    {
      TensorAccessorR<float, 3> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 3> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      for (int i = 0; i < domain.get_dim()-1; i++) {
        assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);
        assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);
      }
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      if (op->momentum > 0.0f) {
        TensorAccessorW<float, 3> accV(
            regions[2], task->regions[2], FID_DATA, ctx, runtime,
            true/*readOutput*/);
        assert(accW.rect == accV.rect);
        v_ptr = accV.ptr;
      }
      break;
    }
    case 4:
    {
      TensorAccessorR<float, 4> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 4> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      for (int i = 0; i < domain.get_dim()-1; i++) {
        assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);
        assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);
      }
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      if (op->momentum > 0.0f) {
        TensorAccessorW<float, 4> accV(
            regions[2], task->regions[2], FID_DATA, ctx, runtime,
            true/*readOutput*/);
        assert(accW.rect == accV.rect);
        v_ptr = accV.ptr;
      }
      break;
    }
    default:
    {
      // Unsupported dims
      assert(false);
    }
  }
  // Step 1: gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    const float* src = w_grad_ptr + i * size;
    apply_add_with_scale<<<GET_BLOCKS(size), CUDA_NUM_THREADS>>>(
        (float*) w_grad_ptr, src, size, 1.0f);
  }
  checkCUDA(cudaDeviceSynchronize());
  // Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS>>>(
      size, op->lr, op->weight_decay, op->momentum, op->nesterov,
      w_grad_ptr, v_ptr, w_ptr);
  checkCUDA(cudaDeviceSynchronize());
}

// ==================================================================
//                        Adam Optimizer
// ==================================================================
__global__
void add_kernel(int count, float scale,
                const float* src,
                float* dst)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    dst[i] += src[i] * scale;
  }
}

__global__
void scale_kernel(int count, float a, float b,
                  float* ptr)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__
void adam_update(int count, float alpha_t,
                 float beta1, float beta2,
                 float weight_decay, float epsilon,
                 const float *WGrad, float *M,
                 float *V, float *W)
{
  // Reference for weight decay
  // https://www.fast.ai/2018/07/02/adam-weight-decay/
  CUDA_KERNEL_LOOP(i, count)
  {
    //W[i] -= weight_decay * alpha_t * W[i];
    //float gt = WGrad[i];
    float gt = WGrad[i] + weight_decay * W[i];
    float mt = beta1 * M[i] + (1 - beta1) * gt;
    float vt = beta2 * V[i] + (1 - beta2) * gt * gt;
    M[i] = mt;
    V[i] = vt;
    W[i] -= alpha_t * mt / (sqrt(vt) + epsilon);
  }
}

__host__
void AdamOptimizer::update_task(const Task* task,
                                const std::vector<PhysicalRegion>& regions,
                                Context ctx, Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const AdamOptimizer* op = (AdamOptimizer*) task->args;
  Domain domain = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  const float *w_grad_ptr = NULL;
  float *w_ptr = NULL, *v_ptr = NULL, *m_ptr = NULL;
  size_t size = 0, num_replicas = 0;
  switch(domain.get_dim()) {
    case 1:
    {
      TensorAccessorR<float, 1> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 1> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      TensorAccessorW<float, 1> accV(
          regions[2], task->regions[2], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      TensorAccessorW<float, 1> accM(
          regions[3], task->regions[3], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      v_ptr = accV.ptr;
      m_ptr = accM.ptr;
      break;
    }
    case 2:
    {
      TensorAccessorR<float, 2> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 2> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      TensorAccessorW<float, 2> accV(
          regions[2], task->regions[2], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      TensorAccessorW<float, 2> accM(
          regions[3], task->regions[3], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      v_ptr = accV.ptr;
      m_ptr = accM.ptr;
      break;
    }
    default:
    {
      // Unsupported dims
      assert(false);
    }
  }
  // Step 1: gather gradients in the first replica
  for (int i = 1; i < num_replicas; i++) {
    const float* src = w_grad_ptr + i * size;
    add_kernel<<<GET_BLOCKS(size), CUDA_NUM_THREADS>>>(
        size, 1.0f, src, (float*)w_grad_ptr);
  }
  checkCUDA(cudaDeviceSynchronize());
  //fprintf(stderr, "alpha = %.8lf alpha_t = %.8lf decay = %.8lf\n",
  //        op->alpha, op->alpha_t, op->weight_decay);
  // Step 2: Adam update
  adam_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS>>>(
      size, op->alpha_t, op->beta1, op->beta2,
      op->weight_decay, op->epsilon,
      w_grad_ptr, m_ptr, v_ptr, w_ptr);
  checkCUDA(cudaDeviceSynchronize());
}

