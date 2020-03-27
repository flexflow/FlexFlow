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
#include "inception.h"
using namespace Legion;

LegionRuntime::Logger::Category log_app("Inceptionv3");

//void parse_input_args(char **argv, int argc, Inceptionv3Config& anConfig);

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
    const int dims[] = {ffConfig.batchSize, 3, 299, 299};
    input = ff.create_tensor<4>(dims, "", DT_FLOAT);
  }
  Tensor label;
  {
    const int dims[] = {ffConfig.batchSize, 1};
    label = ff.create_tensor<2>(dims, "", DT_INT32);
  }
 
//-----------------------------------------------------------------
  Tensor t = ff.conv2d("conv1", input, 32, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d("conv2", t, 32, 3, 3, 1, 1, 0, 0);
  t = ff.conv2d("conv3", t, 64, 3, 3, 1, 1, 1, 1);
  t = ff.pool2d("pool1", t, 3, 3, 2, 2, 0, 0);
  t = ff.conv2d("conv4", t, 80, 1, 1, 1, 1, 0, 0);
  t = ff.conv2d("conv5", t, 192, 3, 3, 1, 1, 1, 1);
  t = ff.pool2d("pool1", t, 3, 3, 2, 2, 0, 0);
  t = InceptionA(ff, t, 32, "ia1_");
  t = InceptionA(ff, t, 64, "ia2_");
  t = InceptionA(ff, t, 64, "ia3_");
  t = InceptionB(ff, t, "ib1_");
  t = InceptionC(ff, t, 128, "ic1_");
  t = InceptionC(ff, t, 160, "ic2_");
  t = InceptionC(ff, t, 160, "ic3_");
  t = InceptionC(ff, t, 192, "ic4_");
  t = InceptionD(ff, t, "id1_");
  t = InceptionE(ff, t, "ie1_");
  t = InceptionE(ff, t, "ie1_");
  t = ff.pool2d("pool1", t, 8, 8, 1, 1, 0, 0, POOL_AVG);
  t = ff.flat("flat", t);
  t = ff.dense("linear1",t, 1000);
  t = ff.softmax("softmax", t, label);
//-----------------------------------------------------------------
  ff.optimizer = new SGDOptimizer(&ff, 0.01f);

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
    int iterations = data_loader.num_samples / ffConfig.batchSize;
 
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
  num_samples = 256 * ff.config.workersPerNode * ff.config.numNodes;
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
