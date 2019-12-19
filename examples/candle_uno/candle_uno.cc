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
#include <string>

using namespace Legion;
using namespace std;

LegionRuntime::Logger::Category log_app("DLRM");

void parse_input_args(char **argv, int argc, CandleConfig& apConfig);

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
  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  // Data Loader
  DataLoader data_loader(ff, candle_config, all_inputs, label);
  ff.init_layers();

  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ff_config.epochs; epoch++) {
    ff.reset_metrics();
    for (int iter = 0; iter < ff_config.iterations; iter++) {
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
  }
}

DataLoader::DataLoader(FFModel& ff,
                       const CandleConfig& candle,
                       const std::vector<Tensor>& _all_inputs,
                       Tensor _label)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgumentMap argmap;
  IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(""));
  for (size_t i = 0; i < _all_inputs.size(); i++) {
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
      RegionRequirement(_all_inputs[i].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, _all_inputs[i].region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Labels
  {
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string("")));
    launcher.add_region_requirement(
      RegionRequirement(_label.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, _label.region));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

void register_custom_tasks()
{
  // Load Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs (Random)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Inputs (Random) Task");
  }
  // Load Label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Label (Random)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Label (Random) Task");
  }
}
