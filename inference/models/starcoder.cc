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

#include "starcoder.h"

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void STARCODER::create_starcoder_model(
    FFModel &ff,
    std::string const &model_config_file_path,
    std::string const &weight_file_path,
    InferenceMode mode,
    GenerationConfig generationConfig,
    bool use_full_precision) {
  // do not apply cpu offload in beam search model.
  STARCODERConfig startcoder_config(model_config_file_path);
  startcoder_config.print();

  if (ff.config.tensor_parallelism_degree >
          startcoder_config.num_attention_heads ||
      startcoder_config.num_attention_heads %
              ff.config.tensor_parallelism_degree !=
          0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;
  std::vector<int> axes = {0};

  Tensor input;
  Tensor position_input;
  ff.set_position_offset(0);
  {
    assert(startcoder_config.max_num_tokens <= BatchConfig::MAX_NUM_TOKENS);
    int const token_dims[] = {BatchConfig::MAX_NUM_TOKENS, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
    position_input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token;

  if (use_full_precision) {
    token = ff.embedding(input,
                         startcoder_config.vocab_size,
                         startcoder_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_FLOAT,
                         NULL,
                         embed_init);
  } else {
    token = ff.embedding(input,
                         startcoder_config.vocab_size,
                         startcoder_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_HALF,
                         NULL,
                         embed_init);
  }

  Layer *embedding = ff.layers.back();
  weights_layers.emplace("transformer_wte_weight", embedding);

  Tensor positional_embedding;
  if (use_full_precision) {
    positional_embedding =
        ff.embedding(position_input,
                     startcoder_config.max_position_embeddings,
                     startcoder_config.hidden_size,
                     AGGR_MODE_NONE,
                     DT_FLOAT,
                     NULL,
                     embed_init);
  } else {
    positional_embedding =
        ff.embedding(position_input,
                     startcoder_config.max_position_embeddings,
                     startcoder_config.hidden_size,
                     AGGR_MODE_NONE,
                     DT_HALF,
                     NULL,
                     embed_init);
  }
  Layer *pos_embedding = ff.layers.back();
  weights_layers.emplace("transformer_wpe_weight", pos_embedding);

  Tensor hidden_states = ff.add(token, positional_embedding);

  for (int i = 0; i < startcoder_config.num_hidden_layers; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);
    // step 1: attention
    Tensor ln_1 = ff.layer_norm(
        hidden_states, axes, true, startcoder_config.layer_norm_epsilon);
    Layer *layer_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_ln_1_weight",
                           layer_norm);

    Tensor mha;
    switch (mode) {
      case INC_DECODING_MODE: {
        mha = ff.inc_multiquery_self_attention(
            ln_1,
            startcoder_config.hidden_size,
            startcoder_config.num_attention_heads,
            1,
            startcoder_config.hidden_size /
                startcoder_config.num_attention_heads,
            startcoder_config.hidden_size /
                startcoder_config.num_attention_heads,
            startcoder_config.dropout_p, /*dropout*/
            true,                        /*bias*/
            false,                       /*add_bias_kv*/
            false,                       /*add_zero_attn*/
            DT_NONE,                     /*data_type*/
            nullptr,                     /*kernel_initializer*/
            false                        /*apply_rotary_embedding*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }
    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);
    Tensor residual = ff.add(hidden_states, mha);

    Tensor l2_norm = ff.layer_norm(
        residual, axes, true, startcoder_config.layer_norm_epsilon);
    Layer *l2_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_ln_2_weight",
                           l2_layer);

    // mlp
    Tensor c_fc = ff.dense(
        l2_norm, startcoder_config.intermediate_size, AC_MODE_NONE, true);
    Layer *c_fc_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_mlp_c_fc_weight",
                           c_fc_layer);
    c_fc = ff.gelu(c_fc);

    Tensor c_proj =
        ff.dense(c_fc, startcoder_config.hidden_size, AC_MODE_NONE, true);
    Layer *c_proj_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_mlp_c_proj_weight",
                           c_proj_layer);

    hidden_states = ff.add(residual, c_proj);
  }
  // final normalization and linear
  Tensor ln_f = ff.layer_norm(
      hidden_states, axes, true, startcoder_config.layer_norm_epsilon);
  Layer *final_norm = ff.layers.back();
  weights_layers.emplace("transformer_ln_f_weight", final_norm);

  Tensor lm_head =
      ff.dense(ln_f, startcoder_config.vocab_size, AC_MODE_NONE, false);
  Layer *final_linear = ff.layers.back();
  weights_layers.emplace("lm_head_weight", final_linear);

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(lm_head, -1);
    //output = ff.beam_top_k(softmax, startcoder_config.max_beam_width, false);
    output = ff.argmax(softmax, /*beam_Search*/ true);
  } else {
    // Tensor softmax = ff.softmax(dense, -1);
    if (generationConfig.do_sample) {
      lm_head = ff.scalar_truediv(lm_head, generationConfig.temperature, false);
      Tensor softmax = ff.softmax(lm_head, -1);
      output = ff.sampling(softmax, generationConfig.topp);
    } else {
       //output = ff.arg_top_k(lm_head, /*k=*/1, false);
       output = ff.argmax(lm_head, /*beam_Search*/ false);
    }
  }

  InferenceManager *im = InferenceManager::get_inference_manager();
  // Compile the model
  std::cout << "------start compile ----------" << std::endl;
  im->compile_model_and_allocate_buffer(&ff);
  FileDataLoader fileloader("",
                            weight_file_path,
                            startcoder_config.num_attention_heads,
                            1,
                            startcoder_config.hidden_size,
                            startcoder_config.hidden_size /
                                startcoder_config.num_attention_heads,
                            ff.config.tensor_parallelism_degree);
  fileloader.load_weights(&ff, weights_layers, use_full_precision);
  std::cout << "------load weight finished----------" << std::endl;

  // init operators
  im->init_operators_inference(&ff);
}

}; // namespace FlexFlow
