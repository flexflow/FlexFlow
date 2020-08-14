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
#include "flexflow_c.h"
using namespace Legion;

LegionRuntime::Logger::Category log_app("AlexNet");

//void parse_input_args(char **argv, int argc, AlexNetConfig& anConfig);

class DataLoader {
public:
  DataLoader(flexflow_model_t ff, flexflow_tensor_t input, flexflow_tensor_t label);
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
  flexflow_config_t ffconfig;
  ffconfig = flexflow_config_create();

  flexflow_config_parse_args_default(ffconfig);
  log_app.print("C API batchSize(%d) workersPerNodes(%d) numNodes(%d)",
    flexflow_config_get_batch_size(ffconfig), flexflow_config_get_workers_per_node(ffconfig), flexflow_config_get_num_nodes(ffconfig));
  
  flexflow_model_t ffmodel = flexflow_model_create(ffconfig);
  FFModel *ff = static_cast<FFModel *>(ffmodel.impl);

  //Tensor input;
  flexflow_tensor_t input;
  {
    const int dims[] = {flexflow_config_get_batch_size(ffconfig), 3, 229, 229};
    //input = ff->create_tensor<4>(dims, "", DT_FLOAT);
    input = flexflow_tensor_4d_create(ffmodel, dims, "", DT_FLOAT, true);
  }
  flexflow_tensor_t label;
  {
    const int dims[] = {flexflow_config_get_batch_size(ffconfig), 1};
    label = flexflow_tensor_2d_create(ffmodel, dims, "", DT_INT32, true);
  }
  // Add layers
  flexflow_tensor_t t0 = input;
  flexflow_tensor_t ts[2];
  ts[0] = flexflow_model_add_conv2d(ffmodel, "conv1", t0, 64, 11, 11, 4, 4, 2, 2, AC_MODE_NONE, true);
  ts[1] = flexflow_model_add_conv2d(ffmodel, "conv1", t0, 64, 11, 11, 4, 4, 2, 2, AC_MODE_NONE, true);
  flexflow_tensor_t t1 = flexflow_model_add_concat(ffmodel, "concat", 2, ts, 1);
  flexflow_tensor_t t2 = flexflow_model_add_pool2d(ffmodel, "pool1", t1, 3, 3, 2, 2, 0, 0, POOL_MAX, AC_MODE_NONE);
  flexflow_tensor_t t3 = flexflow_model_add_conv2d(ffmodel, "conv2", t2, 192, 5, 5, 1, 1, 2, 2, AC_MODE_NONE, true);
  flexflow_tensor_t t4 = flexflow_model_add_pool2d(ffmodel, "pool2", t3, 3, 3, 2, 2, 0, 0, POOL_MAX, AC_MODE_NONE);
  flexflow_tensor_t t5 = flexflow_model_add_conv2d(ffmodel, "conv3", t4, 384, 3, 3, 1, 1, 1, 1, AC_MODE_NONE, true);
  flexflow_tensor_t t6 = flexflow_model_add_conv2d(ffmodel, "conv4", t5, 256, 3, 3, 1, 1, 1, 1, AC_MODE_NONE, true);
  flexflow_tensor_t t7 = flexflow_model_add_conv2d(ffmodel, "conv5", t6, 256, 3, 3, 1, 1, 1, 1, AC_MODE_NONE, true);
  flexflow_tensor_t t8 = flexflow_model_add_pool2d(ffmodel, "pool3", t7, 3, 3, 2, 2, 0, 0, POOL_MAX, AC_MODE_NONE);
  flexflow_tensor_t t9 = flexflow_model_add_flat(ffmodel, "flat", t8);
  flexflow_tensor_t t10 = flexflow_model_add_dense_with_default_initializer(ffmodel, "linear1", t9, 4096, AC_MODE_RELU, true);
  flexflow_tensor_t t11 = flexflow_model_add_dense_with_default_initializer(ffmodel, "linear2", t10, 4096, AC_MODE_RELU, true);
  flexflow_tensor_t t12 = flexflow_model_add_dense_with_default_initializer(ffmodel, "linear3", t11, 1000, AC_MODE_NONE, true);
  flexflow_tensor_t t13 = flexflow_model_add_softmax(ffmodel, "softmax", t12, label);
  //Tensor *tmp_t = static_cast<Tensor *>(t12.impl); 
  //Tensor t = *tmp_t;
  //Tensor t = input;
  //t = ff->conv2d("conv1", t, 64, 11, 11, 4, 4, 2, 2);
  //t = ff->pool2d("pool1", t, 3, 3, 2, 2, 0, 0);
  //t = ff->conv2d("conv2", t, 192, 5, 5, 1, 1, 2, 2);
  //t = ff->pool2d("pool2", t, 3, 3, 2, 2, 0, 0);
  //t = ff->conv2d("conv3", t, 384, 3, 3, 1, 1, 1, 1);
  //t = ff->conv2d("conv4", t, 256, 3, 3, 1, 1, 1, 1);
  //t = ff->conv2d("conv5", t, 256, 3, 3, 1, 1, 1, 1);
  //t = ff->pool2d("pool3", t, 3, 3, 2, 2, 0, 0);
  //t = ff->flat("flat", t);
  //t = ff->linear("lienar1", t, 4096, AC_MODE_RELU/*relu*/);
  //t = ff->linear("linear2", t, 4096, AC_MODE_RELU/*relu*/);
  //t = ff->linear("linear3", t, 1000);
  //t = ff.softmax("softmax", t);
  flexflow_sgd_optimizer_t optimizer = flexflow_sgd_optimizer_create(ffmodel, 0.01f, 0, false, 0);
  flexflow_model_set_sgd_optimizer(ffmodel, optimizer);
  //SGDOptimizer *sgd_opt = static_cast<SGDOptimizer *>(optimizer.impl);
  //ff->optimizer = sgd_opt;
  // Data Loader
  DataLoader data_loader(ffmodel, input, label);
  flexflow_model_init_layers(ffmodel);
  //ff->init_layers();
  //Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < flexflow_config_get_epochs(ffconfig); epoch++) {

    //data_loader.reset();
    ff->reset_metrics();
    int iterations = 8192 / flexflow_config_get_batch_size(ffconfig);
 
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
      flexflow_model_forward(ffmodel);
      flexflow_model_zero_gradients(ffmodel);
      flexflow_model_backward(ffmodel);
      flexflow_model_update(ffmodel);
      //ff->forward();
      //ff->zero_gradients();
      //ff.backward();
      //ff.update();
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
         8192 * flexflow_config_get_epochs(ffconfig) / run_time);
  
  flexflow_tensor_destroy(input);
  flexflow_tensor_destroy(label);   
  flexflow_tensor_destroy(ts[0]);
  flexflow_tensor_destroy(ts[1]);    
  flexflow_tensor_destroy(t1);
  flexflow_tensor_destroy(t2);
  flexflow_tensor_destroy(t3);
  flexflow_tensor_destroy(t4);
  flexflow_tensor_destroy(t5);
  flexflow_tensor_destroy(t6);
  flexflow_tensor_destroy(t7);
  flexflow_tensor_destroy(t8);
  flexflow_tensor_destroy(t9);
  flexflow_tensor_destroy(t10);
  flexflow_tensor_destroy(t11);
  flexflow_tensor_destroy(t12);
  flexflow_tensor_destroy(t13);
  flexflow_sgd_optimizer_destroy(optimizer);
  flexflow_model_destroy(ffmodel);
  flexflow_config_destroy(ffconfig);
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

DataLoader::DataLoader(flexflow_model_t ff_c, flexflow_tensor_t input_c, flexflow_tensor_t label_c)
{
  FFModel *ff = static_cast<FFModel *>(ff_c.impl);
  Tensor *input = static_cast<Tensor *>(input_c.impl);
  Tensor *label = static_cast<Tensor *>(label_c.impl);
  DataLoader(*ff, *input, *label);
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