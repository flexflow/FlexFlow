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
#include <fstream>
#include <sstream>
#include <string>
using namespace Legion;
using namespace FlexFlow;

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  fprintf(stderr,
          "batchSize(%d) workersPerNodes(%d) numNodes(%d)\n",
          ffConfig.batchSize,
          ffConfig.workersPerNode,
          ffConfig.numNodes);
  FFModel ff(ffConfig);

  std::vector<int> hidden_dims = {
      8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};
  Tensor input1, input2;
  {
    int const dims[] = {ffConfig.batchSize, 1024};
    input1 = ff.create_tensor<2>(dims, DT_FLOAT);
    input2 = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  Tensor t1 = input1, t2 = input2;
  for (size_t i = 0; i < hidden_dims.size(); i++) {
    int const dims[] = {hidden_dims[i], t1->dims[0]};
    ActiMode acti_mode =
        (i + 1 == hidden_dims.size()) ? AC_MODE_NONE : AC_MODE_RELU;
    t1 = ff.dense(t1, hidden_dims[i], acti_mode, false);
    t2 = ff.dense(t2, hidden_dims[i], acti_mode, false);
  }
  Tensor t = ff.add(t1, t2);
  t = ff.softmax(t);
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);
  ff.init_operators();
  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    ff.reset_metrics();
    int iterations = 128;
    for (int iter = 0; iter < iterations; iter++) {
      // runtime->begin_trace(ctx, 111 /*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      // ff.backward();
      // ff.update();
      // runtime->end_trace(ctx, 111 /*trace_id*/);
    }
  }
  // End timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n",
         run_time,
         ffConfig.batchSize * 128 * ffConfig.epochs / run_time);
}

void FlexFlow::register_custom_tasks() {}
