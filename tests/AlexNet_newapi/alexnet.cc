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

class DataLoader {
public:
  DataLoader(FFModel& ff, Tensor input, Tensor label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
public:
  int num_samples;
};

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
  Tensor label;
  {
    const int dims[] = {ffConfig.batchSize, 1};
    label = ff.create_tensor<2>(dims, "", DT_INT32);
  }
  
  Tensor t, ts[2];
    
  // Add layers
  Conv2D *conv1_0 = ff.conv2d("conv1", 3, 64, 11, 11, 4, 4, 2, 2);
  Conv2D *conv1_1 = ff.conv2d("conv1", 3, 64, 11, 11, 4, 4, 2, 2);
  Pool2D *pool1 = ff.pool2d("pool1", 3, 3, 2, 2, 0, 0);
  Conv2D *conv2 = ff.conv2d("conv2", 128, 192, 5, 5, 1, 1, 2, 2);
  Pool2D *pool2 = ff.pool2d("pool2", 3, 3, 2, 2, 0, 0);
  Conv2D *conv3 = ff.conv2d("conv3", 192, 384, 3, 3, 1, 1, 1, 1);
  Conv2D *conv4 = ff.conv2d("conv4", 384, 256, 3, 3, 1, 1, 1, 1);
  Conv2D *conv5 = ff.conv2d("conv5", 256, 256, 3, 3, 1, 1, 1, 1);
  Pool2D *pool3 = ff.pool2d("pool3", 3, 3, 2, 2, 0, 0);
  Flat *flat = ff.flat("flat");
  Linear *linear1 = ff.dense("lienar1", 256*6*6, 4096, AC_MODE_RELU/*relu*/);
  Linear *linear2 = ff.dense("linear2", 4096, 4096, AC_MODE_RELU/*relu*/);
  Linear *linear3 = ff.dense("linear3", 4096, 1000);
  
  ts[0] = conv1_0->init_inout(ff, input);
  ts[1] = conv1_1->init_inout(ff, input);
  t = ff.concat("concat", 2, ts, 1/*axis*/);
  t = pool1->init_inout(ff, t);
  t = conv2->init_inout(ff, t);
  t = pool2->init_inout(ff, t);
  t = conv3->init_inout(ff, t);
  t = conv4->init_inout(ff, t);
  t = conv5->init_inout(ff, t);
  t = pool3->init_inout(ff, t);
  t = flat->init_inout(ff, t);
  t = linear1->init_inout(ff, t);
  t = linear2->init_inout(ff, t);
  t = linear3->init_inout(ff, t);
  t = ff.softmax("softmax", t, label);
  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  ff.compile();
  // Data Loader
  DataLoader data_loader(ff, input, label);
  ff.init_layers();
  
  //Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
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
      //ff.forward();
      {
        conv1_0->forward(ff);
        conv1_1->forward(ff);
        ff.layers[2]->forward(ff);
        pool1->forward(ff);
        conv2->forward(ff);
        pool2->forward(ff);
        conv3->forward(ff);
        conv4->forward(ff);
        conv5->forward(ff);
        pool3->forward(ff);
        flat->forward(ff);
        linear1->forward(ff);
        linear2->forward(ff);
        linear3->forward(ff);
        ff.layers[14]->forward(ff);
      }
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
}

void register_custom_tasks()
{
  // Load Input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Inputs Task");
  }
}

DataLoader::DataLoader(FFModel& ff,
                       Tensor input, Tensor label)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = 0;
  log_app.print("Use random dataset...");
  num_samples = 256 * 10 * ff.config.workersPerNode * ff.config.numNodes;
  log_app.print("Number of random samples = %d\n", num_samples);
  // Init input
  {
    IndexSpaceT<4> task_is = IndexSpaceT<4>(ff.get_or_create_task_is(4, ""));
    ArgumentMap argmap;
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, input.region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Init label
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    ArgumentMap argmap;
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
        RegionRequirement(label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, label.region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

void DataLoader::load_input(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime)
{
  printf("CheckPoint#1\n");
}
