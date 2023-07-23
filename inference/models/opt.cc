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

#include "opt.h"

namespace FlexFlow {

using namespace Legion;

void OPT::create_opt_model(FFModel &ff,
                           InferenceManager &im,
                           std::string const &model_config_file_path,
                           std::string const &weight_file_path,
                           InferenceMode mode,
                           bool use_full_precision) {
  Config opt_config(model_config_file_path);
  opt_config.printConfig();
  //---------------------- parallelization setup work ----------------------
  int num_devices = ff.config.workersPerNode * ff.config.numNodes;
  int num_transformer_layers = opt_config.num_hidden_layers;
  assert(num_transformer_layers % ff.config.pipeline_parallelism_degree == 0);
  int num_layers_per_pp_block =
      num_transformer_layers / ff.config.pipeline_parallelism_degree;
  int num_devices_per_data_parallelism_line =
      num_devices / ff.config.data_parallelism_degree;

  // std::cout << "dp: " << ff.config.data_parallelism_degree
  //           << " tp: " << ff.config.tensor_parallelism_degree
  //           << " pp: " << ff.config.pipeline_parallelism_degree << std::endl;
  // std::cout << "num_devices: " << num_devices << std::endl;
  // std::cout << "num_transformer_layers: " << num_transformer_layers
  //           << std::endl;
  // std::cout << "num_devices_per_data_parallelism_line: "
  //           << num_devices_per_data_parallelism_line << std::endl;
  // std::cout << "num layers: " << opt_config.num_hidden_layers << std::endl;

  std::unordered_map<std::string, Layer *> weights_layers;

  //------------------------------ build the model --------------------------
  Tensor input;
  Tensor position_input;
  {
    int const token_dims[] = {BatchConfig::MAX_NUM_TOKENS, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
    position_input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  std::vector<int> axes = {0};

  Tensor token;
  if (use_full_precision) {
    token = ff.embedding(input,
                         opt_config.vocab_size,
                         opt_config.word_embed_proj_dim,
                         AGGR_MODE_NONE,
                         DT_FLOAT,
                         NULL,
                         embed_init);
  } else {
    token = ff.embedding(input,
                         opt_config.vocab_size,
                         opt_config.word_embed_proj_dim,
                         AGGR_MODE_NONE,
                         DT_HALF,
                         NULL,
                         embed_init);
  }

  Layer *embedding = ff.layers.back();
  weights_layers.emplace("embed_tokens_weight", embedding);

  Tensor positional_embedding;
  if (use_full_precision) {
    positional_embedding = ff.embedding(position_input,
                                        opt_config.max_position_embeddings,
                                        opt_config.hidden_size,
                                        AGGR_MODE_NONE,
                                        DT_FLOAT,
                                        NULL,
                                        embed_init);
  } else {
    positional_embedding = ff.embedding(position_input,
                                        opt_config.max_position_embeddings,
                                        opt_config.hidden_size,
                                        AGGR_MODE_NONE,
                                        DT_HALF,
                                        NULL,
                                        embed_init);
  }
  Layer *pos_embedding = ff.layers.back();
  weights_layers.emplace("embed_positions_weight", pos_embedding);

  Tensor residual = ff.add(token, positional_embedding);

  for (int i = 0; i < opt_config.num_hidden_layers; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);

    // 125m, 1.7B, ..., 175B applies layer norm BEFORE attention,
    // 350m applies layer norm AFTER attention
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#LL324C1-L325C1
    // this version is before normalization

    Tensor hidden_states = ff.layer_norm(
        residual, axes, opt_config.layer_norm_elementwise_affine, 1e-05);
    Layer *self_attn_layer_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_attention_layer_norm_weight",
                           self_attn_layer_norm);

    Tensor mha;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multihead_self_attention(
            hidden_states,
            opt_config.hidden_size,
            opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            0.0f,
            true,
            false,
            false,
            DT_NONE, /*data_type*/
            NULL,
            false,
            /*scaling query*/ true,
            /*sacling factor*/
            pow((opt_config.hidden_size / opt_config.num_attention_heads),
                -0.5),
            /*qk_prod_scaling*/ false);
        break;
      }
      case TREE_VERIFY_MODE: {
        mha = ff.inc_multihead_self_attention_verify(
            hidden_states,
            opt_config.hidden_size,
            opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            0.0f,
            true,
            false,
            false,
            DT_NONE, /*data_type*/
            NULL,
            false,
            /*scaling query*/ true,
            /*sacling factor*/
            pow((opt_config.hidden_size / opt_config.num_attention_heads),
                -0.5),
            /*qk_prod_scaling*/ false);
        break;
      }
      case INC_DECODING_MODE: {
        mha = ff.inc_multihead_self_attention(
            hidden_states,
            opt_config.hidden_size,
            opt_config.num_attention_heads,
            opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            0.0f,
            true,
            false,
            false,
            DT_NONE, /*data_type*/
            NULL,
            false,
            /*scaling query*/ true,
            /*sacling factor*/
            pow((opt_config.hidden_size / opt_config.num_attention_heads),
                -0.5),
            /*qk_prod_scaling*/ false);
        break;
      }
      default: {
        assert(false);
      }
    }

    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);

    Tensor added = ff.add(mha, residual);

    Tensor final_norm = ff.layer_norm(
        added, axes, opt_config.layer_norm_elementwise_affine, 1e-05);
    Layer *final_layer_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_final_layer_norm_weight",
                           final_layer_norm);

    //--------linear fc1 fc2 ----------
    Tensor fc1 = ff.dense(final_norm, opt_config.ffn_dim, AC_MODE_NONE, true);
    Layer *fc1_linear = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_fc1_weight",
                           fc1_linear);
    Tensor activation = ff.relu(fc1, false);

    Tensor fc2 =
        ff.dense(activation, opt_config.hidden_size, AC_MODE_NONE, true);
    Layer *fc2_linear = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_fc2_weight",
                           fc2_linear);
    residual = ff.add(added, fc2);
  }

  // final
  Tensor all_final_norm = ff.layer_norm(
      residual, axes, opt_config.layer_norm_elementwise_affine, 1e-05);
  Layer *all_final_norm_layer = ff.layers.back();
  weights_layers.emplace("final_layer_norm_weight", all_final_norm_layer);

  Tensor lm_head =
      ff.dense(all_final_norm, opt_config.vocab_size, AC_MODE_NONE, false);
  Layer *lm_head_layer = ff.layers.back();
  weights_layers.emplace("embed_tokens_weight_lm_head", lm_head_layer);

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(lm_head, -1);
    output = ff.beam_top_k(softmax, opt_config.max_beam_width, false);
  } else {
    output = ff.arg_top_k(lm_head, /*k=*/1, false);
  }

  //------------------- compile the model --------------------------------
  std::cout << "------start compile ----------" << std::endl;
  im.compile_model_and_allocate_buffer(&ff);
  FileDataLoader fileloader("",
                            weight_file_path,
                            opt_config.num_attention_heads,
                            opt_config.num_attention_heads,
                            opt_config.hidden_size,
                            opt_config.hidden_size /
                                opt_config.num_attention_heads);
  fileloader.load_weights(&ff, weights_layers, use_full_precision);
  std::cout << "------finished loading weights----------" << std::endl;
  im.init_operators_inference(&ff);
}

}; // namespace FlexFlow
