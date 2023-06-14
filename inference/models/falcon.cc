/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "falcon.h"

namespace FlexFlow {

using namespace Legion;

void FALCON::create_falcon_model(FFModel &ff,
                                 InferenceManager &im,
                                 std::string const &model_config_file_path,
                                 std::string const &weight_file_path,
                                 int num_pipeline_stages,
                                 InferenceMode mode,
                                 bool use_full_precision) {
  Config falcon_config(model_config_file_path);
  falcon_config.printConfig();
  //------------------------------compute machine views ------------------
  int num_devices = ff.config.workersPerNode * ff.config.numNodes;
  std::vector<MachineView> machine_views;
  for (int i = 0; i < num_devices; i++) {
    MachineView view;
    view.device_type = MachineView::GPU;
    view.ndims = 1;
    view.dim[0] = 1;
    view.stride[0] = 0;
    view.start_device_id = i;
    machine_views.push_back(view);
  }

  std::unordered_map<Tensor, std::vector<MachineView>> mapping;
  std::unordered_map<std::string, Layer *> weights_layers;

  Tensor input;
  {
    assert(falcon_config.max_num_tokens <= BatchConfig::MAX_NUM_TOKENS);
    int const token_dims[] = {BatchConfig::MAX_NUM_TOKENS, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }
  mapping[input].push_back(machine_views[0]);

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token;
  std::vector<int> axes = {0};

  if (use_full_precision) {
    token = ff.embedding(input,
                         falcon_config.vocab_size,
                         falcon_config.dim,
                         AGGR_MODE_NONE,
                         DT_FLOAT,
                         NULL,
                         embed_init);
  } else {
    token = ff.embedding(input,
                         falcon_config.vocab_size,
                         falcon_config.dim,
                         AGGR_MODE_NONE,
                         DT_HALF,
                         NULL,
                         embed_init);
  }

  Layer *embedding = ff.layers.back();
  weights_layers.emplace("tok_embeddings_weight", embedding);

  int num_transformer_layers = falcon_config.n_layers;
  int num_transformer_layers_per_stage =
      (num_transformer_layers + num_pipeline_stages - 1) / num_pipeline_stages;

  for (int i = 0; i < num_transformer_layers; i++) {
    // step 1: attention
    Tensor att_norm = ff.layer_norm(token, axes, true, falcon_config.norm_eps);
    Layer *attention_norm = ff.layers.back();

    if (i % num_transformer_layers_per_stage == 0) {
      // Map att_norm to the next GPU
      // since the size of att_norm is minimum across
      // all tensors
      mapping[att_norm].push_back(
          machine_views[i / num_transformer_layers_per_stage]);
    }

    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_input_layernorm_weight",
                           attention_norm);
    Tensor mha;
    switch (mode) {
      case INC_DECODING_MODE: {
        mha = ff.inc_multiquery_self_attention(
            att_norm,
            falcon_config.dim,
            falcon_config.n_heads,
            falcon_config.dim / falcon_config.n_heads,
            falcon_config.dim / falcon_config.n_heads,
            0.0f,    /*dropout*/
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr  /*kernel_initializer*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }
    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_self_attention_dense_weight",
                           attention_layer);

    Tensor dense_h_to_4h =
        ff.dense(att_norm, falcon_config.dim * 4, AC_MODE_NONE, false);
    Layer *dense_h_to_4h_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_mlp_dense_h_to_4layers_weight",
                           dense_h_to_4h_layer);
    dense_h_to_4h = ff.gelu(dense_h_to_4h);
    Tensor mlp_output =
        ff.dense(dense_h_to_4h, falcon_config.dim, AC_MODE_NONE, false);
    Layer *dense_4h_to_h_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_mlp_dense_4h_to_layers_weight",
                           dense_4h_to_h_layer);

    token = ff.add(token, mha);
    token = ff.add(token, mlp_output);
  }
  // final normalization and linear
  Tensor ln_f = ff.layer_norm(token, axes, true, falcon_config.norm_eps);
  Layer *ln_f_layer = ff.layers.back();
  weights_layers.emplace("ln_f_weight", ln_f_layer);

  Tensor lm_head =
      ff.dense(ln_f, falcon_config.vocab_size, AC_MODE_NONE, false);
  Layer *lm_head_layer = ff.layers.back();
  weights_layers.emplace("lm_head_weight", lm_head_layer);

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(lm_head, -1);
    output = ff.beam_top_k(softmax, falcon_config.max_beam_width, false);
  } else {
    output = ff.arg_top_k(lm_head, /*k=*/1, false);
  }

  // Compile the model
  std::cout << "------start compile ----------" << std::endl;
  im.compile_model_and_allocate_buffer(&ff, mapping);
  FileDataLoader fileloader("",
                            weight_file_path,
                            falcon_config.n_heads,
                            falcon_config.dim,
                            falcon_config.dim / falcon_config.n_heads);
  fileloader.load_weights(&ff, weights_layers);
  std::cout << "------load weight finished----------" << std::endl;

  // init operators
  im.init_operators_inference(&ff);
}

}; // namespace FlexFlow
