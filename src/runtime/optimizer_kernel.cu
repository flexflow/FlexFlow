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
    if (momentum > 0) {
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
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const SGDOptimizer* op = (SGDOptimizer*) task->args;
  Domain domain = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  const float *w_grad_ptr;
  float *w_ptr, *v_ptr;
  size_t size = 0, num_replicas = 0;
  switch(domain.get_dim()) {
    case 1:
    {
      TensorAccessorR<float, 2> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 1> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      TensorAccessorW<float, 1> accV(
          regions[2], task->regions[2], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      assert(accW.rect == accV.rect);
      for (int i = 0; i < domain.get_dim(); i++) {
        assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);
        assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);
      }
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      v_ptr = accV.ptr;
      break;
    }
    case 2:
    {
      TensorAccessorR<float, 3> accWGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      TensorAccessorW<float, 2> accW(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      TensorAccessorW<float, 2> accV(
          regions[2], task->regions[2], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      assert(accW.rect == accV.rect);
      for (int i = 0; i < domain.get_dim(); i++) {
        assert(accW.rect.lo[i] == accWGrad.rect.lo[i]);
        assert(accW.rect.hi[i] == accWGrad.rect.hi[i]);
      }
      size = accW.rect.volume();
      assert(accWGrad.rect.volume() % accW.rect.volume() == 0);
      num_replicas = accWGrad.rect.volume() / accW.rect.volume();
      w_grad_ptr = accWGrad.ptr;
      w_ptr = accW.ptr;
      v_ptr = accV.ptr;
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
  // Step 2: SGD update
  sgd_update<<<GET_BLOCKS(size), CUDA_NUM_THREADS>>>(
      size, op->lr, op->weight_decay, op->momentum, op->nesterov,
      w_grad_ptr, v_ptr, w_ptr);
  checkCUDA(cudaDeviceSynchronize());
}

