/* Copyright 2020 Stanford
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

#include "flexflow/model.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

using namespace Legion;

__global__
void sparse_categorical_crossentropy_loss_backward(
    float *logit_grad,
    const int *label,
    coord_t num_samples,
    coord_t num_classes,
    const int k)
{
  CUDA_KERNEL_LOOP(i, num_samples)
  {
    int label_idx = label[i/k];
    logit_grad[i * num_classes + label_idx] -= 1.0f;
  }
}

__global__
void categorical_crossentropy_loss_backward(
    float *logit_grad,
    const float *logit,
    const float *label,
    coord_t num_elements)
{
  CUDA_KERNEL_LOOP(i, num_elements)
  {
    logit_grad[i] = logit[i] - label[i];
  }
}

__global__
void mean_squared_error_avg_loss_backward(
    float *logit_grad,
    const float *logit,
    const float *label,
    coord_t num_elements)
{
  CUDA_KERNEL_LOOP(i, num_elements)
  {
    logit_grad[i] = logit[i] - label[i];
  }
}

__host__
void Loss::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
__host__
void Loss::backward_task_with_dim(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Loss* loss = (Loss*) task->args;
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (loss->loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    //sparse_categorical_crossentropy has label of dim: (batch_size, 1)
    TensorAccessorW<float, NDIM> acc_logit_grad(
        regions[0], task->regions[0], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    TensorAccessorR<float, NDIM> acc_logit(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    TensorAccessorR<int, NDIM> acc_label(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
    // assertion the outter-most dim is replica dim and replica degree is 1
    assert(acc_logit.rect.hi[NDIM-1] == acc_logit.rect.lo[NDIM-1]);
    int num_samples = acc_logit.rect.hi[NDIM-2] - acc_logit.rect.lo[NDIM-2] + 1;
    int num_classes = acc_logit.rect.volume() / num_samples;
    assert(acc_logit_grad.rect == acc_logit.rect);
    int k = 1;
    if(loss->repl_labels) {
      k = (acc_logit.rect.hi[NDIM-1]-acc_logit.rect.lo[NDIM-1]+1) /
        (acc_label.rect.hi[NDIM-1]-acc_label.rect.lo[NDIM-1]+1);
    }
    for (int i = 1; i < NDIM-1; i++) {
      assert(acc_label.rect.hi[i] == acc_logit.rect.hi[i]);
      assert(acc_label.rect.lo[i] == acc_logit.rect.lo[i]);
    }
    assert(k*(acc_label.rect.hi[NDIM-1]-acc_label.rect.lo[NDIM-1]+1)
      == acc_logit.rect.hi[NDIM-1]-acc_logit.rect.lo[NDIM-1]+1);
    assert(acc_label.rect.lo[0] == acc_label.rect.hi[0]);
    checkCUDA(cudaMemcpy(acc_logit_grad.ptr, acc_logit.ptr,
                         acc_logit.rect.volume() * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    sparse_categorical_crossentropy_loss_backward<<<GET_BLOCKS(num_samples), CUDA_NUM_THREADS, 0, stream>>>(
        acc_logit_grad.ptr, acc_label.ptr, num_samples, num_classes, k);
    // Scale logit gradients by op->scale_factor
    scale_kernel<<<GET_BLOCKS(acc_logit_grad.rect.volume()), CUDA_NUM_THREADS, 0, stream>>>(
        acc_logit_grad.ptr, acc_logit_grad.rect.volume(), 0, loss->scale_factor*k);
  } else {
    if(loss->repl_labels) assert(false && "Loss not yet supported for aggr_spec.");
    TensorAccessorW<float, NDIM> acc_logit_grad(
        regions[0], task->regions[0], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    TensorAccessorR<float, NDIM> acc_logit(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    TensorAccessorR<float, NDIM> acc_label(
        regions[2], task->regions[2], FID_DATA, ctx, runtime);
    // other loss require label and logit have identical shape
    assert(acc_logit.rect == acc_label.rect);
    assert(acc_logit_grad.rect == acc_logit.rect);
    // assertion the outter-most dim is replica dim and replica degree is 1
    assert(acc_logit.rect.hi[NDIM-1] == acc_logit.rect.lo[NDIM-1]);
    int num_samples = acc_label.rect.hi[NDIM-2] - acc_label.rect.lo[NDIM-2] + 1;
    int num_channels = acc_logit.rect.volume() / num_samples;
    if (loss->loss_type == LOSS_CATEGORICAL_CROSSENTROPY) {
      categorical_crossentropy_loss_backward<<<GET_BLOCKS(acc_logit.rect.volume()), CUDA_NUM_THREADS, 0, stream>>>(
          acc_logit_grad.ptr, acc_logit.ptr, acc_label.ptr,
          acc_logit.rect.volume());
      // Scale logit gradients by loss->scale_factor
      scale_kernel<<<GET_BLOCKS(acc_logit_grad.rect.volume()), CUDA_NUM_THREADS, 0, stream>>>(
          acc_logit_grad.ptr, acc_logit_grad.rect.volume(), 0, loss->scale_factor);
    } else if (loss->loss_type == LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE) {
      mean_squared_error_avg_loss_backward<<<GET_BLOCKS(acc_logit.rect.volume()), CUDA_NUM_THREADS, 0, stream>>>(
          acc_logit_grad.ptr, acc_logit.ptr, acc_label.ptr,
          acc_logit.rect.volume());
      // Scale logit gradients by loss->scale_factor
      scale_kernel<<<GET_BLOCKS(acc_logit_grad.rect.volume()), CUDA_NUM_THREADS, 0, stream>>>(
          acc_logit_grad.ptr, acc_logit_grad.rect.volume(), 0, loss->scale_factor);
    } else {
      fprintf(stderr, "Unsupported loss --- report this error to the FlexFlow developers\n");
      assert(false);
    }
  }
}

}; // namespace FlexFlow
