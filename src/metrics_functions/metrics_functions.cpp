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

#include <hip/hip_runtime.h>
#include "flexflow/model.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::PhysicalRegion;

const float LOG_MIN_VALUE = 0.00000001f;

__global__
void update_metrics_sparse_label_kernel(
    const float* logits,
    const int* labels,
    PerfMetrics* perf,
    const Metrics metrics,
    int num_samples,
    int num_classes)
{
  CUDA_KERNEL_LOOP(b, num_samples)
  {
    if (metrics.measure_accuracy) {
      float max_val = -1.0f;
      int my_label = -1;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b*num_classes+i];
        if (my_logit > max_val) {
          max_val = my_logit;
          my_label = i;
        }
      }
      assert(my_label >= 0);
      atomicAdd(&(perf->train_all), 1);
      if (labels[b] == my_label)
        atomicAdd(&(perf->train_correct), 1);
    }
    if (metrics.measure_sparse_categorical_crossentropy) {
      float my_logit = max(logits[b*num_classes+labels[b]], LOG_MIN_VALUE);
      atomicAdd(&(perf->sparse_cce_loss), -log(my_logit));
    }
    if (metrics.measure_mean_squared_error
    || metrics.measure_root_mean_squared_error
    || metrics.measure_mean_absolute_error)
    {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b*num_classes+i];
        float my_label = (labels[b] == i) ? 1.0f : 0.0f;
        mse += (my_logit - my_label) * (my_logit - my_label);
        mae += abs(my_logit - my_label);
      }
      if (metrics.measure_mean_squared_error)
        atomicAdd(&(perf->mse_loss), mse);
      if (metrics.measure_root_mean_squared_error)
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      if (metrics.measure_mean_absolute_error)
        atomicAdd(&(perf->mae_loss), mae);
    }
  }
}

__global__
void update_metrics_label_kernel(
    const float* logits,
    const float* labels,
    PerfMetrics* perf,
    const Metrics metrics,
    int num_samples,
    int num_classes)
{
  CUDA_KERNEL_LOOP(b, num_samples)
  {
    atomicAdd(&(perf->train_all), 1);
    if (metrics.measure_accuracy) {
      if (num_classes == 1) {
        // accuracy does not make sense when num_classes = 1
        // we just return 100%
        atomicAdd(&(perf->train_all), 1);
        atomicAdd(&(perf->train_correct), 1);
      } else {
        float max_val = 0.0f;
        int my_label = -1, true_label = -1;
        for (int i = 0; i < num_classes; i++) {
          if (my_label == -1 || logits[b*num_classes+i] > max_val) {
            max_val = logits[b*num_classes+i];
            my_label = i;
          }
          if (labels[b*num_classes+i] > 0.9f) {
            assert(true_label == -1);
            true_label = i;
          }
        }
        assert(my_label >= 0);
        assert(true_label >= 0);
        if (true_label == my_label)
          atomicAdd(&(perf->train_correct), 1);
      }
    }
    if (metrics.measure_categorical_crossentropy) {
      float cce = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        if (labels[b*num_classes+i] > 0.0f) {
          float my_logit = max(logits[b*num_classes+i], LOG_MIN_VALUE);
          cce += labels[b*num_classes+i] * -log(my_logit);
        }
      }
      atomicAdd(&(perf->cce_loss), cce);
    }
    if (metrics.measure_mean_squared_error
    || metrics.measure_root_mean_squared_error
    || metrics.measure_mean_absolute_error)
    {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float diff = logits[b*num_classes+i] - labels[b*num_classes+i];
        mse += diff * diff;
        mae += abs(diff);
      }
      if (metrics.measure_mean_squared_error)
        atomicAdd(&(perf->mse_loss), mse);
      if (metrics.measure_root_mean_squared_error)
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      if (metrics.measure_mean_absolute_error)
        atomicAdd(&(perf->mae_loss), mae);
    }
  }
}

__host__
PerfMetrics Metrics::compute_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime)
{
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return compute_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  PerfMetrics invalid;
  return invalid;
}

template<int NDIM>
__host__
PerfMetrics Metrics::compute_task_with_dim(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Metrics* me = (Metrics*) task->args;
  PerfMetrics* perf;
  PerfMetrics perf_zc;
  checkCUDA(hipMalloc(&perf, sizeof(PerfMetrics)));
  checkCUDA(hipMemcpy(perf, &perf_zc, sizeof(PerfMetrics), hipMemcpyHostToDevice));

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (me->loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    TensorAccessorR<float, NDIM> acc_logit(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<int, NDIM> acc_label(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    // assume that the leading dim is replica dim
    assert(acc_logit.rect.hi[NDIM-1] == acc_logit.rect.lo[NDIM-1]);
    int num_samples = acc_logit.rect.hi[NDIM-2] - acc_logit.rect.lo[NDIM-2] + 1;
    int num_classes = acc_logit.rect.volume() / num_samples;
    for (int i = 1; i < NDIM; i++) {
      assert(acc_label.rect.hi[i] == acc_logit.rect.hi[i]);
      assert(acc_label.rect.lo[i] == acc_logit.rect.lo[i]);
    }
    assert(acc_label.rect.lo[0] == acc_label.rect.hi[0]);
    // Cannot measure categorical_crossentropy w/ sparse labels
    // Use measure_sparse_categorical_crossentropy instead
    assert(!me->measure_categorical_crossentropy);
    hipLaunchKernelGGL(update_metrics_sparse_label_kernel, GET_BLOCKS(num_samples), CUDA_NUM_THREADS, 0, stream, 
        acc_logit.ptr, acc_label.ptr, perf, *me, num_samples, num_classes);
  } else {
    TensorAccessorR<float, NDIM> acc_logit(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<float, NDIM> acc_label(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    // other loss require label and logit have identical shape
    assert(acc_logit.rect == acc_label.rect);
    // assume that the leading dim is replica dim
    assert(acc_logit.rect.hi[NDIM-1] == acc_logit.rect.lo[NDIM-1]);
    int num_samples = acc_logit.rect.hi[NDIM-2] - acc_logit.rect.lo[NDIM-2] + 1;
    int num_classes = acc_logit.rect.volume() / num_samples;
    // Use CUDA_NUM_THREADS may result in out of resources so we set #threads=256
    hipLaunchKernelGGL(update_metrics_label_kernel, GET_BLOCKS(num_samples), 256, 0, stream, 
      acc_logit.ptr, acc_label.ptr, perf, *me, num_samples, num_classes);
  }
  checkCUDA(hipStreamSynchronize(stream));
  checkCUDA(hipMemcpy(&perf_zc, perf, sizeof(PerfMetrics), hipMemcpyDeviceToHost));
  checkCUDA(hipFree(perf));
  return perf_zc;
}

}; // namespace FlexFlow
