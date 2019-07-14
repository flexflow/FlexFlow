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

LegionRuntime::Logger::Category log_ff("DLRM");

struct DLRMConfig {
  DLRMConfig(void)
  : sparse_feature_size(2), loss_threshold(0.0f) {}
  int sparse_feature_size;
  float loss_threshold;
  std::vector<int> embedding_size, mlp_bot, mlp_top;
};

void parse_input_args(char **argv, int argc, DLRMConfig& apConfig);

Tensor create_mlp(FFModel* model, const Tensor& input,
                  std::vector<int> ln, int sigmoid_layer)
{
  Tensor t = input;
  for (size_t i = 0; i < ln.size()-1; i++) {
    float std_dev = sqrt(2.0f / (ln[i+1] + ln[i]));
    Initializer* weight_init = new NormInitializer(std::rand(), 0, std_dev);
    std_dev = sqrt(2.0f / ln[i+1]);
    Initializer* bias_init = new NormInitializer(std::rand(), 0, std_dev);
    ActiMode activation = i == sigmoid_layer ? AC_MODE_SIGMOID : AC_MODE_RELU;
    t = model->linear("linear", t, ln[i+1], weight_init, bias_init, activation);
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
    Tensor* inputs = malloc(sizeof(Tensor) * (1 + ly.size()));
    inputs[0] = x;
    for (size_t i = 0; i < ly.size(); i++)
      inputs[i+1] = ly[i];
    return model->concat("concat", ly.size() + 1, inputs, 1/*axis*/);
    free(inputs);
  } else {
    assert(false);
  }
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
    parse_input_args(argv, argc, ffConfig, dlrmConfig);
  }

  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);

  std::vector<Tensor> sparse_inputs;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    const int dims[] = {ffConfig.batchSize, 1};
    Tensor input = ff.create_input_tensor<float>(2, dims);
    sparse_inputs.push_back(input);
  }
  Tensor dense_input;
  {
    const int dims[] = {ffCOnfig.batchSize, dlrmConfig.mlp_bot[0]};
    dense_input = ff.create_input_tensor<int>(2, dims);
  }
  Tensor label;
  // Step 1 create dense_mlp
  Tensor x = create_mlp(&ff, dense_input, dlrmConfig.mlp_bot);
  std::vector<Tensor> ly;
  for (size_t i = 0; i < dlrmConfig.embedding_size.size(); i++) {
    int input_dim = dlrmConfig.embedding_size[i];
    int output_dim = dlrmConfig.sparse_feature_size;
    ly.push_back(create_emb(&ff, sparse_inputs[i], input_dim, output_dim);
  }
  Tensor z = interact_features(x, ly);
  Tensor p = create_mlp(z, dlrmConfig.mlp_top);
  if (dlrmConfig.loss_threshold > 0.0f && dlrmConfig.loss_thresdhold < 1.0f) {
    // TODO: implement clamp
    assert(false):
  }
  ff.mse_loss("mse_loss"/*name*/, p, label, "mean"/*reduction*/);
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    for (int batch = 0; batch < ffConfig.batches; batch++) {
      ff.zero_gradients();
      ff.forward();
      ff.backward();
      ff.update();
    }
  }
}

void parse_input_args(char **argv, int argc, DLRMConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if ((!strcmp(argv[i]), "--arch-sparse-feature-size")) {
      config.sparse_feature_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-embedding-size")) {
      std::stringstream ss(std::string(argv[++i])), word;
      while (std::get_line(ss, word, "-")) {
        config.embedding_size.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-bot")) {
      std::stringstream ss(std::string(argv[++i])), word;
      while (std::get_line(ss, word, "-")) {
        config.mlp_bot.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-top")) {
      std::stringstream ss(std::string(argv[++i])), word;
      while (std::get_line(ss, word, "-")) {
        config.mlp_top.push_back(std::stoi(word));
      }
      continue;
    }
  }
}
