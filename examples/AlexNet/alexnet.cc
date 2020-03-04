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

#include "model.h"
using namespace Legion;

LegionRuntime::Logger::Category log_app("AlexNet");

//void parse_input_args(char **argv, int argc, AlexNetConfig& anConfig);

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  FFConfig ffConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
  }
  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);

  Tensor input;
  {
    const int dims[] = {ffConfig.batchSize, 3, 229, 229};
    input = ff.create_tensor<4>(dims, "", DT_FLOAT);
  }
  //Tensor label;
  //{
    //const int dims[] = {ffConfig.batchSize, 1};
    //label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  //}
  // Add layers
  Tensor t = input;
  t = ff.conv2d("conv1", t, 64, 11, 11, 4, 4, 2, 2);
  t = ff.pool2d("pool1", t, 3, 3, 2, 2, 0, 0);
  ff.print_layers();
#if 0
  t = ff.conv2d("conv2", t, 192, 5, 5, 1, 1, 2, 2);
  t = ff.pool2d("pool2", t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d("conv3", t, 384, 3, 3, 1, 1, 1, 1);
  t = ff.conv2d("conv4", t, 256, 3, 3, 1, 1, 1, 1);
  t = ff.conv2d("conv5", t, 256, 3, 3, 1, 1, 1, 1);
  t = ff.pool2d("pool3", t, 3, 3, 2, 2, 0, 0);
  t = ff.flat("flat", t);
  t = ff.linear("lienar1", t, 4096);
  t = ff.linear("linear2", t, 4096);
  t = ff.linear("linear3", t, 1000, AC_MODE_RELU/*relu*/);
  t = ff.softmax("softmax", t);

  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  ff.init_layers();
  //TODO: implement a data loader
  //Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < 0; epoch++) {
  //for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    //data_loader.reset();
    ff.reset_metrics();
    int iterations = 8192 / ffConfig.batchSize;
 
    for (int iter = 0; iter < iterations; iter++) {
      //if (dlrmConfig.dataset_path.length() == 0) {
        // Only load data once for random input
        //if (iter == 0 && epoch == 0)
          //data_loader.next_batch(ff);
      //} else {
        //data_loader.next_batch(ff);
      //}
      if (epoch > 0)
        runtime->begin_trace(ctx, 111/*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      if (epoch > 0)
        runtime->end_trace(ctx, 111/*trace_id*/);
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
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n", run_time,
         8192 * ffConfig.epochs / run_time);
#endif
}

void register_custom_tasks()
{
}
