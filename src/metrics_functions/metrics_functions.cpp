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

#include "flexflow/model.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

float const LOG_MIN_VALUE = 0.00000001f;
int const MASK_TOKEN = -100;

__global__ void update_metrics_sparse_label_kernel(float const *logits,
                                                   int const *labels,
                                                   PerfMetrics *perf,
                                                   const Metrics metrics,
                                                   int num_samples,
                                                   int num_classes) {
  CUDA_KERNEL_LOOP(b, num_samples) {
    if (metrics.measure_accuracy) {
      float max_val = -1.0f;
      int my_label = 0;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b * num_classes + i];
        if (my_logit > max_val) {
          max_val = my_logit;
          my_label = i;
        }
      }
      assert(my_label >= 0);
      if (labels[b] != MASK_TOKEN) {
        atomicAdd(&(perf->train_all), 1);
        if (labels[b] == my_label) {
          atomicAdd(&(perf->train_correct), 1);
        }
      }
    }
    if (metrics.measure_sparse_categorical_crossentropy) {
      if (labels[b] != MASK_TOKEN) {
        float my_logit =
            max(logits[b * num_classes + labels[b]], LOG_MIN_VALUE);
        atomicAdd(&(perf->sparse_cce_loss), -log(my_logit));
      }
    }
    if (metrics.measure_mean_squared_error ||
        metrics.measure_root_mean_squared_error ||
        metrics.measure_mean_absolute_error) {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float my_logit = logits[b * num_classes + i];
        float my_label = (labels[b] == i) ? 1.0f : 0.0f;
        mse += (my_logit - my_label) * (my_logit - my_label);
        mae += abs(my_logit - my_label);
      }
      if (metrics.measure_mean_squared_error) {
        atomicAdd(&(perf->mse_loss), mse);
      }
      if (metrics.measure_root_mean_squared_error) {
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      }
      if (metrics.measure_mean_absolute_error) {
        atomicAdd(&(perf->mae_loss), mae);
      }
    }
  }
}

__global__ void update_metrics_label_kernel(float const *logits,
                                            float const *labels,
                                            PerfMetrics *perf,
                                            const Metrics metrics,
                                            int num_samples,
                                            int num_classes) {
  CUDA_KERNEL_LOOP(b, num_samples) {
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
          if (my_label == -1 || logits[b * num_classes + i] > max_val) {
            max_val = logits[b * num_classes + i];
            my_label = i;
          }
          if (labels[b * num_classes + i] > 0.9f) {
            assert(true_label == -1);
            true_label = i;
          }
        }
        assert(my_label >= 0);
        assert(true_label >= 0);
        if (true_label == my_label) {
          atomicAdd(&(perf->train_correct), 1);
        }
      }
    }
    if (metrics.measure_categorical_crossentropy) {
      float cce = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        if (labels[b * num_classes + i] > 0.0f) {
          float my_logit = max(logits[b * num_classes + i], LOG_MIN_VALUE);
          cce += labels[b * num_classes + i] * -log(my_logit);
        }
      }
      atomicAdd(&(perf->cce_loss), cce);
    }
    if (metrics.measure_mean_squared_error ||
        metrics.measure_root_mean_squared_error ||
        metrics.measure_mean_absolute_error) {
      float mse = 0.0f, mae = 0.0f;
      for (int i = 0; i < num_classes; i++) {
        float diff = logits[b * num_classes + i] - labels[b * num_classes + i];
        mse += diff * diff;
        mae += abs(diff);
      }
      if (metrics.measure_mean_squared_error) {
        atomicAdd(&(perf->mse_loss), mse);
      }
      if (metrics.measure_root_mean_squared_error) {
        atomicAdd(&(perf->rmse_loss), sqrt(mse));
      }
      if (metrics.measure_mean_absolute_error) {
        atomicAdd(&(perf->mae_loss), mae);
      }
    }
  }
}

void Metrics::update_metrics_sparse_label_kernel_wrapper(
    float const *logit_ptr,
    int const *label_ptr,
    Metrics const *me,
    int num_effective_samples,
    int num_classes,
    PerfMetrics &perf_zc) {
  PerfMetrics *perf;
  checkCUDA(hipMalloc(&perf, sizeof(PerfMetrics)));
  checkCUDA(
      hipMemcpy(perf, &perf_zc, sizeof(PerfMetrics), hipMemcpyHostToDevice));

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  hipLaunchKernelGGL(update_metrics_sparse_label_kernel,
                     GET_BLOCKS(num_effective_samples),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     logit_ptr,
                     label_ptr,
                     perf,
                     *me,
                     num_effective_samples,
                     num_classes);
  checkCUDA(hipStreamSynchronize(stream));
  checkCUDA(
      hipMemcpy(&perf_zc, perf, sizeof(PerfMetrics), hipMemcpyDeviceToHost));
  checkCUDA(hipFree(perf));
}

void Metrics::update_metrics_label_kernel_wrapper(float const *logit_ptr,
                                                  float const *label_ptr,
                                                  Metrics const *me,
                                                  int num_samples,
                                                  int num_classes,
                                                  PerfMetrics &perf_zc) {
  PerfMetrics *perf;
  checkCUDA(hipMalloc(&perf, sizeof(PerfMetrics)));
  checkCUDA(
      hipMemcpy(perf, &perf_zc, sizeof(PerfMetrics), hipMemcpyHostToDevice));

  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  hipLaunchKernelGGL(update_metrics_label_kernel,
                     GET_BLOCKS(num_samples),
                     256,
                     0,
                     stream,
                     logit_ptr,
                     label_ptr,
                     perf,
                     *me,
                     num_samples,
                     num_classes);
  checkCUDA(hipStreamSynchronize(stream));
  checkCUDA(
      hipMemcpy(&perf_zc, perf, sizeof(PerfMetrics), hipMemcpyDeviceToHost));
  checkCUDA(hipFree(perf));
}

}; // namespace FlexFlow
