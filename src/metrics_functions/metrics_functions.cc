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

#include "metrics_functions.h"


PerfMetrics::PerfMetrics(void)
: train_all(0), train_correct(0), cce_loss(0.0f), sparse_cce_loss(0.0f),
  mse_loss(0.0f), rmse_loss(0.0f), mae_loss(0.0f)
{}

void PerfMetrics::update(const PerfMetrics& one)
{
  train_all += one.train_all;
  train_correct += one.train_correct;
  cce_loss += one.cce_loss;
  sparse_cce_loss += one.sparse_cce_loss;
  mse_loss += one.mse_loss;
  rmse_loss += one.rmse_loss;
  mae_loss += one.mae_loss;
}

void PerfMetrics::apply_scale(float scale)
{
  cce_loss *= scale;
  sparse_cce_loss *= scale;
  mse_loss *= scale;
  rmse_loss *= scale;
  mae_loss *= scale;
}

void PerfMetrics::print(void)
{
  std::string output = "[Metrics]";
  if (train_all > 0) {
    float accuracy = train_correct * 100.0f / train_all;
    output = output + " accuracy: " + std::to_string(accuracy) + "% ("
           + std::to_string(train_correct) + " / "
           + std::to_string(train_all) + ")";
  }
  if (cce_loss > 0) {
    float avg_cce_loss = cce_loss / train_all;
    output = output + " categorical_crossentropy: " + std::to_string(avg_cce_loss);
  }
  if (sparse_cce_loss > 0) {
    float avg_cce_loss = sparse_cce_loss / train_all;
    output = output + " sparse_categorical_crossentropy: " + std::to_string(avg_cce_loss);
  }
  if (mse_loss > 0) {
    output = output + " mean_squared_error: " + std::to_string(mse_loss / train_all);
  }
  if (rmse_loss > 0) {
    output = output + " root_mean_squared_error: " + std::to_string(rmse_loss / train_all);
  }
  if (mae_loss > 0) {
    output = output + " mean_absolute_error: " + std::to_string(mae_loss / train_all);
  }
  fprintf(stderr, "%s\n", output.c_str());
}

