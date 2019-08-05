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

#include "model.h"
#include <sstream>
#include <string>

using namespace Legion;
using namespace std;

LegionRuntime::Logger::Category log_app("DLRM");

struct CandleConfig {
  CandleConfig(void) {
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
    input_features["cell.rnaseq"] = "cell.rnaseq";
    input_features["drug1.descriptors"] = "drug.descriptors";
    input_features["drug1.fingerprints"] = "drug.fingerprints";
  }
  vector<int> dense_layers, dense_feature_layers;
  map<string, int> feature_shapes;
  map<string, string> input_features;
};

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
  Tensor encoded_inputs[MAX_NUM_INPUTS];
  for (map<string, string>::const_iterator it = input_features.begin();
      it != input_features.end(); it++)
  {
    assert(feature_shapes.find(it->second) != feature_shapes.end());
    int shape = feature_shapes[it->second];
    const int dims[] = {ff_config.batchSize, shape};
    Tensor input = ff.create_tensor<2>(dims, "", DT_FLOAT);
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
  ff.init_layers();
  for (int epoch = 0; epoch < ff_config.epochs; epoch++) {
    for (int iter = 0; iter < ff_config.iterations; iter++) {
      printf("epoch = %d iter = %d\n", epoch, iter);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
    }
  }
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

