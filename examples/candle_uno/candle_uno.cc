/* Copyright 2019 Stanford, LANL
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

#include "candle_uno.h"
#include <sstream>
#include <fstream>
#include <string>

using namespace Legion;
using namespace std;

LegionRuntime::Logger::Category log_app("DLRM");

void parse_input_args(char **argv, int argc, CandleConfig& apConfig);

CandleConfig::CandleConfig(void)
{
  // Set default configurations here
  for (int i = 0; i < 3; i++)
    dense_layers.push_back(1000);
  for (int i = 0; i < 3; i++)
    dense_feature_layers.push_back(1000);
  feature_shapes["dose"] = 1;
  feature_shapes["cell.rnaseq"] = 942;
  feature_shapes["drug.descriptors"] = 5270;
  feature_shapes["drug.fingerprints"] = 2048;
  input_features["dose1"] = "dose";
  input_features["dose2"] = "dose";
  input_features["cell.rnaseq"] = "cell.rnaseq";
  input_features["drug1.descriptors"] = "drug.descriptors";
  input_features["drug1.fingerprints"] = "drug.fingerprints";
  //input_features["drug2.descriptors"] = "drug.descriptors";
  //input_features["drug2.fingerprints"] = "drug.fingerprints";
}

Tensor build_feature_model(FFModel* model, const Tensor& input,
                           const std::vector<int>& dense_layers)
{
  Tensor t = input;
  for (size_t i = 0; i < dense_layers.size(); i++) {
    t = model->dense("dense", t, dense_layers[i], AC_MODE_RELU);
  }
  return t;
}

void print_vector(const std::string& name, const std::vector<int>& vector)
{
  std::ostringstream out;
  for (size_t i = 0; i < vector.size() - 1; i++)
    out << vector[i] << " ";
  if (vector.size() > 0)
    out << vector[vector.size() - 1];
  log_app.print("%s: %s", name.c_str(), out.str().c_str());
}

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  FFConfig ff_config;
  CandleConfig candle_config;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ff_config.parse_args(argv, argc);
    parse_input_args(argv, argc, candle_config);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ff_config.batchSize, ff_config.workersPerNode, ff_config.numNodes);
    print_vector("Dense Layers", candle_config.dense_layers);
    print_vector("Dense Feature Layers", candle_config.dense_feature_layers);
  }
  
  ff_config.lg_ctx = ctx;
  ff_config.lg_hlr = runtime;
  ff_config.field_space = runtime->create_field_space(ctx);
  FFModel ff(ff_config);
  set<string> input_models;
  map<string, string> input_features = candle_config.input_features;
  map<string, int> feature_shapes = candle_config.feature_shapes;
  for (map<string, int>::const_iterator it = feature_shapes.begin();
      it != feature_shapes.end(); it++)
  {
    string fea_type = it->first;
    if (fea_type.find(".") != string::npos) {
      string base_type = fea_type.substr(0, fea_type.find("."));
      if (base_type == "cell" || base_type == "drug")
        input_models.insert(it->first);
    }
  }
  int n = 0;
  std::vector<Tensor> all_inputs;
  Tensor encoded_inputs[MAX_NUM_INPUTS];
  for (map<string, string>::const_iterator it = input_features.begin();
      it != input_features.end(); it++)
  {
    assert(feature_shapes.find(it->second) != feature_shapes.end());
    int shape = feature_shapes[it->second];
    const int dims[] = {ff_config.batchSize, shape};
    Tensor input = ff.create_tensor<2>(dims, "", DT_FLOAT);
    all_inputs.push_back(input);
    if (input_models.find(it->second) != input_models.end()) {
      Tensor encoded = build_feature_model(&ff, input, candle_config.dense_feature_layers);
      encoded_inputs[n++] = encoded;
    } else {
      encoded_inputs[n++] = input;
    }
  }
  Tensor output = ff.concat("concat", n, encoded_inputs, 1/*axis*/);
  for (size_t i = 0; i < candle_config.dense_layers.size(); i++) {
    int out_dim = candle_config.dense_layers[i];
    output = ff.dense("dense", output, out_dim, AC_MODE_RELU);
  }
  output = ff.dense("dense", output, 1);
  Tensor label;
  {
    const int dims[] = {ff_config.batchSize, 1};
    label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  ff.mse_loss("mse_loss", output, label, "average"/*reduction*/);
  // Use SGD Optimizer
  ff.optimizer = new SGDOptimizer(&ff, 0.001f);
  // Data Loader
  DataLoader data_loader(ff, candle_config, all_inputs, label);
  ff.init_layers();

  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ff_config.epochs; epoch++) {
    data_loader.reset();
    ff.reset_metrics();
    int iterations = data_loader.num_samples / ff_config.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      if (candle_config.dataset_path.length() == 0) {
        // Only load data once for random input
        if (iter == 0 && epoch == 0)
          data_loader.next_batch(ff);
      } else {
        data_loader.next_batch(ff);
      }
      //runtime->begin_trace(ctx, 111/*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      //runtime->end_trace(ctx, 111/*trace_id*/);
    }
  }
  runtime->issue_execution_fence(ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = runtime->issue_timing_measurement(ctx, timer);
  future.get_void_result();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n", run_time,
         ff_config.iterations * ff_config.epochs * ff_config.batchSize / run_time);
}

void parse_input_args(char **argv, int argc, CandleConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dense-layers")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.dense_layers.clear();
      while (std::getline(ss, word, '-')) {
        config.dense_layers.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--dense-feature-layers")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.dense_feature_layers.clear();
      while (std::getline(ss, word, '-')) {
        config.dense_feature_layers.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

size_t get_file_size(const std::string& filename)
{
  streampos begin,end;
  ifstream file(filename.c_str(), ios::binary);
  begin = file.tellg();
  file.seekg (0, ios::end);
  end = file.tellg();
  file.close();
  size_t filesize = end - begin;
  printf("filesize(%s) = %zu\n", filename.c_str(), filesize);
  return filesize;
}

DataLoader::DataLoader(FFModel& ff,
                       const CandleConfig& candle,
                       const std::vector<Tensor>& _inputs,
                       Tensor _label)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = 0;
  if (candle.dataset_path == "") {
    log_app.print("Use random dataset...");
    num_samples = 8192;
  } else {
    log_app.print("Start loading dataset from %s", candle.dataset_path.c_str());
    assert(_inputs.size() == candle.input_features.size());
    string dose1_name = candle.dataset_path + "/dose1";
    num_samples = get_file_size(dose1_name) / 4;
    // inputs
    int idx = 0;
    for(map<string, string>::const_iterator it = candle.input_features.begin();
        it != candle.input_features.end(); it++)
    {
      string filename = candle.dataset_path+it->first;
      size_t filesize = get_file_size(filename);
      assert(filesize == (size_t)num_samples * sizeof(float) * _inputs[idx++].adim[0]);
    }
    // labels
    {
      string filename = candle.dataset_path + "/label";
      assert(get_file_size(filename) == (size_t)num_samples * sizeof(float));
    }
  }
  for (size_t i = 0; i < _inputs.size(); i++) {
    batch_inputs.push_back(_inputs[i]);
    const int dims[] = {num_samples, _inputs[i].adim[0]};
    Tensor full_input = ff.create_tensor<2>(dims, "", DT_FLOAT);
    full_inputs.push_back(full_input);
  }
  {
    batch_label = _label;
    const int dims[] = {num_samples, 1};
    full_label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  const CandleConfig* ptr = &candle;
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
      TaskArgument(&ptr, sizeof(CandleConfig*)));
  // regions[0]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region, WRITE_ONLY,
                        EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1-n]: full_inputs
  for (size_t i = 0; i < full_inputs.size(); i++) {
    launcher.add_region_requirement(
        RegionRequirement(full_inputs[i].region, WRITE_ONLY,
                          EXCLUSIVE, full_inputs[i].region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx,
                                     Runtime* runtime)
{
  CandleConfig* candle = *((CandleConfig**)task->args);
  assert(regions.size() == candle->input_features.size() + 1);
  assert(task->regions.size() == regions.size());
  const AccessorWO<float, 2> acc_label(regions[0], FID_DATA);
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float* label_ptr = acc_label.ptr(rect_label.lo);
  int num_samples = rect_label.hi[1] - rect_label.lo[1] + 1;
  if (candle->dataset_path.length() == 0) {
    log_app.print("Start generating random input samples");
    for (size_t i = 0; i < rect_label.volume(); i++)
      label_ptr[i] = ((float)std::rand()) / RAND_MAX - 0.5f;
  } else {
    string filename = candle->dataset_path + "/label";
    log_app.print("Start loading labels from %s", filename.c_str());
    FILE* file = fopen(filename.c_str(), "rb");
    size_t ret = fread(label_ptr, sizeof(float), rect_label.volume(), file);
    assert(ret == rect_label.volume());
    fclose(file);
  }
  int idx = 0;
  for(map<string, string>::const_iterator it = candle->input_features.begin();
      it != candle->input_features.end(); it++, idx++)
  {
    printf("idx = %d\n", idx);
    const AccessorWO<float, 2> acc_input(regions[idx+1], FID_DATA);
    Rect<2> rect_input = runtime->get_index_space_domain(
        ctx, task->regions[idx+1].region.get_index_space());
    assert(acc_input.accessor.is_dense_arbitrary(rect_input));
    float* input_ptr = acc_input.ptr(rect_input.lo);
    assert(num_samples == rect_input.hi[1] - rect_input.lo[1] + 1);
    //int num_features = rect_input.hi[0] - rect_input.lo[0] + 1;
    if (candle->dataset_path.length() == 0) {
      for (size_t j = 0; j < rect_input.volume(); j++) {
        input_ptr[j] = ((float)std::rand()) / RAND_MAX;
      }
    } else {
      string filename = candle->dataset_path + it->first;
      log_app.print("Start loading input feature %s from %s",
                    it->first.c_str(), filename.c_str());
      FILE* file = fopen(filename.c_str(), "rb");
      size_t ret = fread(input_ptr, sizeof(float), rect_input.volume(), file);
      assert(ret == rect_input.volume());
      fclose(file);
      log_app.print("Finish loading input feature %s", it->first.c_str());
    }
  }
}

void DataLoader::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(full_inputs.size() == batch_inputs.size());
  // Load inputs
  for (size_t i = 0; i < batch_inputs.size(); i++) {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(""));
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
                           TaskArgument(&i, sizeof(int)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_inputs[i].region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_inputs[i].region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_inputs[i].part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_inputs[i].region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    std::string pc_name = "";
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(pc_name));
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
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string(pc_name)));
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
  // progress next_index
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
}
