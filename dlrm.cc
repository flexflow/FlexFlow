/* Copyright 2019 Stanford
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

using namespace Legion;

LegionRuntime::Logger::Category log_app("DLRM");

struct DLRMConfig {
  DLRMConfig(void)
  : sparse_feature_size(2), sigmoid_bot(-1), sigmoid_top(-1),
    loss_threshold(0.0f), arch_interaction_op("dot") {}
  int sparse_feature_size, sigmoid_bot, sigmoid_top;
  float loss_threshold;
  std::vector<int> embedding_size, mlp_bot, mlp_top;
  std::string arch_interaction_op;
};

void parse_input_args(char **argv, int argc, DLRMConfig& apConfig);

Tensor create_mlp(FFModel* model, const Tensor& input,
                  std::vector<int> ln, int sigmoid_layer)
{
  Tensor t = input;
  for (int i = 0; i < (int)(ln.size()-1); i++) {
    float std_dev = sqrt(2.0f / (ln[i+1] + ln[i]));
    Initializer* weight_init = new NormInitializer(std::rand(), 0, std_dev);
    std_dev = sqrt(2.0f / ln[i+1]);
    Initializer* bias_init = new NormInitializer(std::rand(), 0, std_dev);
    ActiMode activation = i == sigmoid_layer ? AC_MODE_SIGMOID : AC_MODE_RELU;
    t = model->linear("linear", t, ln[i+1], activation, weight_init, bias_init);
  }
  return t;
}

Tensor create_emb(FFModel* model, const Tensor& input,
                  int input_dim, int output_dim)
{
  float range = sqrt(1.0f / input_dim);
  Initializer* embed_init = new UniformInitializer(std::rand(), -range, range);
  return model->embedding("embedding", input, input_dim, output_dim, embed_init);
}

Tensor interact_features(FFModel* model, const Tensor& x,
                         const std::vector<Tensor>& ly,
                         std::string interaction)
{
  // Currently only support cat
  // TODO: implement dot attention
  if (interaction == "cat") {
    Tensor* inputs = (Tensor*) malloc(sizeof(Tensor) * (1 + ly.size()));
    inputs[0] = x;
    for (size_t i = 0; i < ly.size(); i++)
      inputs[i+1] = ly[i];
    return model->concat("concat", ly.size() + 1, inputs, 1/*axis*/);
    free(inputs);
  } else {
    assert(false);
  }
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
  FFConfig ffConfig;
  // Parse input arguments
  DLRMConfig dlrmConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    parse_input_args(argv, argc, dlrmConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
    print_vector("Embedding Size", dlrmConfig.embedding_size);
    print_vector("MLP Top", dlrmConfig.mlp_top);
    print_vector("MLP Bot", dlrmConfig.mlp_bot);
  }

  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);

  std::vector<Tensor> sparse_inputs;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    const int dims[] = {ffConfig.batchSize, 1};
    Tensor input = ff.create_tensor<2>(dims, "", DT_INT32);
    sparse_inputs.push_back(input);
  }
  Tensor dense_input;
  {
    const int dims[] = {ffConfig.batchSize, dlrmConfig.mlp_bot[0]};
    dense_input = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  Tensor label;
  {
    const int dims[] = {ffConfig.batchSize, 2};
    label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  // Step 1 create dense_mlp
  Tensor x = create_mlp(&ff, dense_input, dlrmConfig.mlp_bot, dlrmConfig.sigmoid_bot);
  std::vector<Tensor> ly;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    int input_dim = dlrmConfig.embedding_size[i];
    int output_dim = dlrmConfig.sparse_feature_size;
    ly.push_back(create_emb(&ff, sparse_inputs[i], input_dim, output_dim));
  }
  Tensor z = interact_features(&ff, x, ly, dlrmConfig.arch_interaction_op);
  Tensor p = create_mlp(&ff, z, dlrmConfig.mlp_top, dlrmConfig.sigmoid_top);
  if (dlrmConfig.loss_threshold > 0.0f && dlrmConfig.loss_threshold < 1.0f) {
    // TODO: implement clamp
    assert(false);
  }
  ff.mse_loss("mse_loss"/*name*/, p, label, "average"/*reduction*/);
  // Use SGD Optimizer
  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  ff.init_layers();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    for (int iter = 0; iter < ffConfig.numIterations; iter++) {
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
    }
  }
}

void parse_input_args(char **argv, int argc, DLRMConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--arch-sparse-feature-size")) {
      config.sparse_feature_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-embedding-size")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      while (std::getline(ss, word, '-')) {
        config.embedding_size.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-bot")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      while (std::getline(ss, word, '-')) {
        config.mlp_bot.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-top")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      while (std::getline(ss, word, '-')) {
        config.mlp_top.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--loss-threshold")) {
      config.loss_threshold = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-top")) {
      config.sigmoid_top = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-bot")) {
      config.sigmoid_bot = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-interaction-op")) {
      config.arch_interaction_op = std::string(argv[++i]);
    }
  }
}
