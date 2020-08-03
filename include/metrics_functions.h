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

#ifndef _FF_METRICS_FUNCTIONS_H_
#define _FF_METRICS_FUNCTIONS_H_

#include "legion.h"
#include "loss_functions.h"

class Tensor;
class FFModel;

class PerfMetrics
{
public:
  PerfMetrics(void);
  void update(const PerfMetrics& one);
  void apply_scale(float scale_factor);
  void print();
public:
  int train_all, train_correct; // measure_accuracy
  float cce_loss; // measure_categorical_crossentropy
  float sparse_cce_loss; // measure_sparse_categorical_crossentropy
  float mse_loss; // measure_mean_squared_error
  float rmse_loss; // measure_root_mean_squared_error
  float mae_loss; // measure_mean_absolute_error
};

class Metrics
{
public:
  Metrics(LossType _loss_type, const std::vector<MetricsType>& metrics);
  static PerfMetrics compute_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
  void compute(FFModel* model, const Tensor* logit, const Tensor* label);
public:
  LossType loss_type;
  bool measure_accuracy;
  bool measure_categorical_crossentropy;
  bool measure_sparse_categorical_crossentropy;
  bool measure_mean_squared_error;
  bool measure_root_mean_squared_error;
  bool measure_mean_absolute_error;
};
#endif
