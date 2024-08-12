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

#include "inception.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace Legion;
using namespace FlexFlow;

Realm::Logger log_app("Inceptionv3");

Tensor InceptionA(FFModel &ff, Tensor input, int pool_features) {
  Tensor t1 = input;
  t1 = ff.conv2d(t1, 64, 1, 1, 1, 1, 0, 0, AC_MODE_RELU);

  Tensor t2 = ff.conv2d(input, 48, 1, 1, 1, 1, 0, 0, AC_MODE_RELU);
  t2 = ff.conv2d(t2, 64, 5, 5, 1, 1, 2, 2, AC_MODE_RELU);

  Tensor t3 = ff.conv2d(input, 64, 1, 1, 1, 1, 0, 0, AC_MODE_RELU);
  t3 = ff.conv2d(t3, 96, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t3 = ff.conv2d(t3, 96, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);

  Tensor t4 = ff.pool2d(input, 3, 3, 1, 1, 1, 1, POOL_AVG);
  t4 = ff.conv2d(t4, pool_features, 1, 1, 1, 1, 0, 0, AC_MODE_RELU);

  Tensor concat[4];
  concat[0] = t1;
  concat[1] = t2;
  concat[2] = t3;
  concat[3] = t4;
  Tensor output = ff.concat(4, concat, 1);

  return output;
}

Tensor InceptionB(FFModel &ff, Tensor input) {
  Tensor t1 = ff.conv2d(input, 384, 3, 3, 2, 2, 0, 0);
  Tensor t2 = ff.conv2d(input, 64, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(t2, 96, 3, 3, 1, 1, 1, 1);
  t2 = ff.conv2d(t2, 96, 3, 3, 2, 2, 0, 0);
  Tensor t3 = ff.pool2d(input, 3, 3, 2, 2, 0, 0);
  Tensor concat[3];
  concat[0] = t1;
  concat[1] = t2;
  concat[2] = t3;
  Tensor output = ff.concat(3, concat, 1);
  return output;
}

Tensor InceptionC(FFModel &ff, Tensor input, int channels) {
  Tensor t1 = ff.conv2d(input, 192, 1, 1, 1, 1, 0, 0);
  Tensor t2 = ff.conv2d(input, channels, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(t2, channels, 1, 7, 1, 1, 0, 3);
  t2 = ff.conv2d(t2, 192, 7, 1, 1, 1, 3, 0);
  Tensor t3 = ff.conv2d(input, channels, 1, 1, 1, 1, 0, 0);
  t3 = ff.conv2d(t3, channels, 7, 1, 1, 1, 3, 0);
  t3 = ff.conv2d(t3, channels, 1, 7, 1, 1, 0, 3);
  t3 = ff.conv2d(t3, channels, 7, 1, 1, 1, 3, 0);
  t3 = ff.conv2d(t3, 192, 1, 7, 1, 1, 0, 3);
  Tensor t4 = ff.pool2d(input, 3, 3, 1, 1, 1, 1, POOL_AVG);
  t4 = ff.conv2d(t4, 192, 1, 1, 1, 1, 0, 0);
  Tensor concat[4];
  concat[0] = t1;
  concat[1] = t2;
  concat[2] = t3;
  concat[3] = t4;
  Tensor output = ff.concat(4, concat, 1);
  return output;
}

Tensor InceptionD(FFModel &ff, Tensor input) {
  Tensor t1 = ff.conv2d(input, 192, 1, 1, 1, 1, 0, 0);
  t1 = ff.conv2d(t1, 320, 3, 3, 2, 2, 0, 0);
  Tensor t2 = ff.conv2d(input, 192, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(t2, 192, 1, 7, 1, 1, 0, 3);
  t2 = ff.conv2d(t2, 192, 7, 1, 1, 1, 3, 0);
  t2 = ff.conv2d(t2, 192, 3, 3, 2, 2, 0, 0);
  Tensor t3 = ff.pool2d(input, 3, 3, 2, 2, 0, 0);
  Tensor concat[3];
  concat[0] = t1;
  concat[1] = t2;
  concat[2] = t3;
  Tensor output = ff.concat(3, concat, 1);
  return output;
}

Tensor InceptionE(FFModel &ff, Tensor input) {
  Tensor t1 = ff.conv2d(input, 320, 1, 1, 1, 1, 0, 0);
  Tensor t2i = ff.conv2d(input, 384, 1, 1, 1, 1, 0, 0);
  Tensor t2 = ff.conv2d(t2i, 384, 1, 3, 1, 1, 0, 1);
  Tensor t3 = ff.conv2d(t2i, 384, 3, 1, 1, 1, 1, 0);
  Tensor t3i = ff.conv2d(input, 448, 1, 1, 1, 1, 0, 0);
  t3i = ff.conv2d(t3i, 384, 3, 3, 1, 1, 1, 1);
  Tensor t4 = ff.conv2d(t3i, 384, 1, 3, 1, 1, 0, 1);
  Tensor t5 = ff.conv2d(t3i, 384, 3, 1, 1, 1, 1, 0);
  Tensor t6 = ff.pool2d(input, 3, 3, 1, 1, 1, 1, POOL_AVG);
  t6 = ff.conv2d(t6, 192, 1, 1, 1, 1, 0, 0);
  Tensor concat[6];
  concat[0] = t1;
  concat[1] = t2;
  concat[2] = t3;
  concat[3] = t4;
  concat[4] = t5;
  concat[5] = t6;
  Tensor output = ff.concat(6, concat, 1);
  return output;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  /* { */
  /* const InputArgs &command_args = HighLevelRuntime::get_input_args(); */
  /* char **argv = command_args.argv; */
  /* int argc = command_args.argc; */
  /* parse_input_args(argv, argc, inceptionConfig); */
  log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
                ffConfig.batchSize,
                ffConfig.workersPerNode,
                ffConfig.numNodes);
  /* } */
  FFModel ff(ffConfig);

  Tensor input;
  {
    int const dims[] = {ffConfig.batchSize, 3, 299, 299};
    input = ff.create_tensor<4>(dims, DT_FLOAT);
  }
  // Tensor label;
  //{
  //   const int dims[] = {ffConfig.batchSize, 1};
  //   label = ff.create_tensor<2>(dims, DT_INT32);
  // }

  //-----------------------------------------------------------------
  Tensor t = ff.conv2d(input, 32, 3, 3, 2, 2, 0, 0, AC_MODE_RELU);
  t = ff.conv2d(t, 32, 3, 3, 1, 1, 0, 0, AC_MODE_RELU);
  t = ff.conv2d(t, 64, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d(t, 80, 1, 1, 1, 1, 0, 0, AC_MODE_RELU);
  t = ff.conv2d(t, 192, 3, 3, 1, 1, 1, 1, AC_MODE_RELU);
  t = ff.pool2d(t, 3, 3, 2, 2, 0, 0);

  t = InceptionA(ff, t, 32);
  t = InceptionA(ff, t, 64);
  t = InceptionA(ff, t, 64);
  t = InceptionB(ff, t);
  t = InceptionC(ff, t, 128);
  t = InceptionC(ff, t, 160);
  t = InceptionC(ff, t, 160);
  t = InceptionC(ff, t, 192);
  t = InceptionD(ff, t);
  t = InceptionE(ff, t);
  t = InceptionE(ff, t);
  t = ff.pool2d(t, 8, 8, 1, 1, 0, 0, POOL_AVG);
  t = ff.flat(t);
  t = ff.dense(t, 10);
  t = ff.softmax(t);
  //-----------------------------------------------------------------
  Optimizer *optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  // Data Loader
  /* DataLoader data_loader(ff, inceptionConfig, input, ff.label_tensor); */
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
    /* data_loader.reset(); */
    ff.reset_metrics();
    /* int iterations = data_loader.num_samples / ffConfig.batchSize; */
    int iterations = 128;

    for (int iter = 0; iter < iterations; iter++) {
      /* if (inceptionConfig.dataset_path.length() == 0) { */
      /*   // Only load data once for random input */
      /*   if (iter == 0 && epoch == 0) */
      /*     data_loader.next_batch(ff); */
      /* } else { */
      /*   data_loader.next_batch(ff); */
      /* } */
      // if (epoch > 0) {
      //   runtime->begin_trace(ctx, 111 /*trace_id*/);
      // }
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      // if (epoch > 0) {
      //   runtime->end_trace(ctx, 111 /*trace_id*/);
      // }
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
         8192 * ffConfig.epochs / run_time);
}

void FlexFlow::register_custom_tasks() {}
