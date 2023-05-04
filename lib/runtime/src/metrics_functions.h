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
#include "kernels/perf_metrics.h"

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_metrics;

class Metrics {
public:
  Metrics() = delete;
  Metrics(LossType, std::vector<MetricsType> const &);
public:
  LossType loss_type;
  bool measure_accuracy;
  bool measure_categorical_crossentropy;
  bool measure_sparse_categorical_crossentropy;
  bool measure_mean_squared_error;
  bool measure_root_mean_squared_error;
  bool measure_mean_absolute_error;
};

TaskInvocation compute_metrics(Metrics const &,
                               parallel_tensor_guid_t const &logit,
                               parallel_tensor_guid_t const &label);

TaskInvocation update_metrics(Metrics const &,
                              parallel_tensor_guid_t const &logit,
                              parallel_tensor_guid_t const &label);

template <> void register_task<METRICS_COMP_TASK_ID>();
template <> void register_task<UPDATE_METRICS_TASK_ID>();

PerfMetrics compute_metrics_task(Legion::Task const *,
                                 std::vector<Legion::PhysicalRegion> const &,
                                 Legion::Context, 
                                 Legion::Runtime *);
PerfMetrics update_metrics_task(Legion::Task const *,
                                std::vector<Legion::PhysicalRegion> const &,
                                Legion::Context, 
                                Legion::Runtime *);

}

VISITABLE_STRUCT(::FlexFlow::Metrics, 
                 loss_type, 
                 measure_accuracy, 
                 measure_categorical_crossentropy, 
                 measure_sparse_categorical_crossentropy,
                 measure_mean_squared_error,
                 measure_root_mean_squared_error,
                 measure_mean_absolute_error);

#endif
