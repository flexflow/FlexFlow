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
#include "moe.h"
#include <sstream>
#include <fstream>
#include <string>

#define NUM_SAMPLES 60000
#define TRAIN_SAMPLES 60000
#define TEST_SAMPLES 0
#define MNIST_DIMS 28*28
#define CIFAR_DIMS 3*32*32
#define DATA_DIMS MNIST_DIMS
#define OUT_DIM 10


using namespace Legion;

LegionRuntime::Logger::Category log_app("MoE");


void parse_input_args(char **argv, int argc, MoeConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  FFConfig ffConfig;
  MoeConfig moeConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, moeConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
  }
  FFModel ff(ffConfig);

  Tensor input;
  {
    const int dims[] = {ffConfig.batchSize, DATA_DIMS};
    input = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  //Tensor label;
  //{
  //  const int dims[] = {ffConfig.batchSize, 1};
  //  label = ff.create_tensor<2>(dims, DT_INT32);
  //}

//-----------------------------------------------------------------

  int num_exp = 5;
  int num_select = 2;
  float alpha = 2.0f; // factor overhead tensor size for imbalance
  float lambda = 0.16f; // multiplier for load balance term


  // MoE model
  Tensor gate_preds = ff.dense(input, num_exp, AC_MODE_RELU);
  Tensor topK_output[2];
  ff.top_k(gate_preds, topK_output, num_select, false);
  ff.cache(topK_output[1], TRAIN_SAMPLES / ffConfig.batchSize);


  Tensor exp_tensors[num_exp];
  ff.group_by(input, topK_output[1], exp_tensors, num_exp, alpha);

  Tensor agg_inputs[num_exp+3];
  agg_inputs[0] = ff.softmax(topK_output[0]); // gate preds
  agg_inputs[1] = topK_output[1]; // gate assign
  agg_inputs[2] = gate_preds; // full gate preds
  for(int i = 0; i < num_exp; i++) {
    agg_inputs[i+3] = ff.dense(exp_tensors[i], OUT_DIM, AC_MODE_RELU);
    agg_inputs[i+3] = ff.softmax(agg_inputs[i+3]);
  }

  Tensor coop_output = ff.aggregate(agg_inputs, num_exp, lambda);
  ff.get_metrics();
  Tensor final_pred = ff.aggregate_spec(agg_inputs, num_exp, lambda);

//-----------------------------------------------------------------

  Optimizer* optimizer = new SGDOptimizer(&ff, 0.001f);
  std::vector<MetricsType> metrics;
  metrics.push_back(METRICS_ACCURACY);
  metrics.push_back(METRICS_SPARSE_CATEGORICAL_CROSSENTROPY);
  ff.compile(optimizer, LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics);

  // Data Loader
  DataLoader data_loader(ff, moeConfig, input, ff.label_tensor);
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
    data_loader.reset();
    ff.reset_metrics();
    int iterations = TRAIN_SAMPLES / ffConfig.batchSize;

    for (int iter = 0; iter < iterations; iter++) {
      data_loader.next_batch(ff);
      // if (epoch > 0)
      //    runtime->begin_trace(ctx, 111/*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      // if (epoch > 0)
      //    runtime->end_trace(ctx, 111/*trace_id*/);
    }

    ff.reset_metrics();
    iterations = TEST_SAMPLES / ffConfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      data_loader.next_batch(ff);
      ff.forward();
      ff.backward();
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
         TRAIN_SAMPLES * ffConfig.epochs / run_time);
}


DataLoader::DataLoader(FFModel& ff, const MoeConfig& moe,
                       Tensor input, Tensor label)
{
  num_samples = NUM_SAMPLES;

  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  // Create full input
  {
    batch_input = input;
    const int dims[] = {NUM_SAMPLES, input.adim[0]};
    full_input = ff.create_tensor<2>(dims, DT_FLOAT);
  }
  // Create full label
  {
    batch_label = label;
    const int dims[] = {NUM_SAMPLES, label.adim[0]};
    full_label = ff.create_tensor<2>(dims, DT_INT32);
  }

  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  const MoeConfig* ptr = &moe;
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
      TaskArgument(&ptr, sizeof(MoeConfig*)));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input.region, WRITE_ONLY,
                        EXCLUSIVE, full_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region, WRITE_ONLY,
                        EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);

  runtime->execute_task(ctx, launcher);
  reset();
  next_batch(ff);
}

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}


// =================================================
//                    Load MNIST
// =================================================

/* NOTE: Download files from http://yann.lecun.com/exdb/mnist/, unpack to
this directory (Flexflow/examples/cpp/mixture_of_experts) */


void read_cifar100(float* input_ptr, int* label_ptr) {
  std::ifstream file;
  file.open("train.bin", std::ios::in | std::ios::binary | std::ios::ate);
  if (!file) {
      std::cout << "Error opening CIFAR100 train data file" << std::endl;
      return;
  }

  file.seekg(0, std::ios::beg);

  // each sample: <1 x coarse label><1 x fine label><3072 x pixel>
  for(std::size_t i = 0; i < NUM_SAMPLES; i++) {
    unsigned char temp = 0;
    file.read((char*)&temp, sizeof(temp)); // coarse label, skip
    file.read((char*)&temp, sizeof(temp));
    label_ptr[i] = temp;
    for(std::size_t j = 0; j < 3072; ++j) {
      file.read((char*)&temp, sizeof(temp));
      input_ptr[i*3072 + j] = (float)temp/255.0f;
    }
  }

  file.close();
}



int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void read_mnist(float* input_ptr, int* label_ptr)
{
  // read inputs
  std::ifstream input("train-images-idx3-ubyte", std::ios::binary);
  if (input.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    input.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    input.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    input.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    input.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);

    for(int i = 0; i < number_of_images; i++) {
      for(int r = 0; r < n_rows; r++) {
        for(int c = 0; c < n_cols; c++) {
          unsigned char temp=0;
          input.read((char*)&temp,sizeof(temp));
          input_ptr[i*n_rows*n_cols + r*n_cols + c] = (float)temp/255.0f;
        }
      }
    }
  }
  else {
    assert(false);
  }

  // read labels
  std::ifstream labels("train-labels-idx1-ubyte", std::ios::binary);
  if (labels.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    labels.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    labels.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    for(int i = 0; i < number_of_images; i++) {
      unsigned char temp = 0;
      labels.read((char*)&temp, sizeof(temp));
      label_ptr[i] = temp;
    }
  }
  else {
    assert(false);
  }
}


void DataLoader::load_entire_dataset(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime* runtime)
{
  //const MoeConfig* conf = *((MoeConfig**)task->args);
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());

  // get input and label pointer
  const AccessorWO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<int, 2> acc_label(regions[1], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float* input_ptr = acc_input.ptr(rect_input.lo);
  int* label_ptr = acc_label.ptr(rect_label.lo);

  read_mnist(input_ptr, label_ptr);
  log_app.print("finish loading data\n");
}


void DataLoader::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load input
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_label.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

void DataLoader::reset()
{
  next_index = 0;
}

void register_custom_tasks()
{
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
}
