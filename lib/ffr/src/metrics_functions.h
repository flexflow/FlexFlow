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

#ifndef _FF_METRICS_FUNCTIONS_H_
#define _FF_METRICS_FUNCTIONS_H_

#include "legion.h"
#include "loss_functions.h"

namespace FlexFlow {

class FFModel;
class Metrics;

class PerfMetrics {
public:
  PerfMetrics(void);
  void update(PerfMetrics const &one);
  void apply_scale(float scale_factor);
  void print(Metrics const *m);

public:
  int train_all, train_correct; // measure_accuracy
  float cce_loss;               // measure_categorical_crossentropy
  float sparse_cce_loss;        // measure_sparse_categorical_crossentropy
  float mse_loss;               // measure_mean_squared_error
  float rmse_loss;              // measure_root_mean_squared_error
  float mae_loss;               // measure_mean_absolute_error
  double start_time;
};

class Metrics {
public:
  Metrics(LossType _loss_type, std::vector<MetricsType> const &metrics);
  static PerfMetrics
      compute_task(Legion::Task const *task,
                   std::vector<Legion::PhysicalRegion> const &regions,
                   Legion::Context ctx,
                   Legion::Runtime *runtime);
  template <int NDIM>
  static PerfMetrics
      compute_task_with_dim(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void update_metrics_sparse_label_kernel_wrapper(float const *logit_ptr,
                                                         int const *label_ptr,
                                                         Metrics const *me,
                                                         int num_samples,
                                                         int num_classes,
                                                         PerfMetrics &perf_zc);
  static void update_metrics_label_kernel_wrapper(float const *logit_ptr,
                                                  float const *label_ptr,
                                                  Metrics const *me,
                                                  int num_samples,
                                                  int num_classes,
                                                  PerfMetrics &perf_zc);
  void compute(FFModel *model,
               const ParallelTensor logit,
               const ParallelTensor label);
  template <int NDIM>
  void compute_with_dim(FFModel *model,
                        const ParallelTensor logit,
                        const ParallelTensor label);

public:
  LossType loss_type;
  bool measure_accuracy;
  bool measure_categorical_crossentropy;
  bool measure_sparse_categorical_crossentropy;
  bool measure_mean_squared_error;
  bool measure_root_mean_squared_error;
  bool measure_mean_absolute_error;
};

}; // namespace FlexFlow
#endif
