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
    // assert(startcoder_config.max_num_tokens <= BatchConfig::MAX_NUM_TOKENS);
    int const token_dims[] = {
        (mode == TREE_VERIFY_MODE || mode == BEAM_SEARCH_MODE)
            ? BatchConfig::max_verify_tokens_per_batch()
            : BatchConfig::max_tokens_per_batch(),
        1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
    position_input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token = ff.embedding(input,
                              startcoder_config.vocab_size,
                              startcoder_config.hidden_size,
                              AGGR_MODE_NONE,
                              use_full_precision ? DT_FLOAT : DT_HALF,
                              NULL,
                              embed_init,
                              "transformer_wte");

  Tensor positional_embedding =
      ff.embedding(position_input,
                   startcoder_config.max_position_embeddings,
                   startcoder_config.hidden_size,
                   AGGR_MODE_NONE,
                   use_full_precision ? DT_FLOAT : DT_HALF,
                   NULL,
                   embed_init,
                   "transformer_wpe");

  Tensor residual = nullptr, c_proj = nullptr;
  Tensor res_ln_outputs[2] = {nullptr, nullptr};

  for (int i = 0; i < startcoder_config.num_hidden_layers; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);

    // step 1: attention
    ff.residual_layer_norm(
        (i == 0) ? token : residual,
        (i == 0) ? positional_embedding : c_proj,
        nullptr,
        res_ln_outputs,
        false,
        axes,
        true,
        startcoder_config.layer_norm_epsilon,
        true,
        DT_NONE,
        std::string("layers_" + std::to_string(i) + "_ln_1").c_str());
    Tensor hidden_states = res_ln_outputs[0];
    Tensor ln_1 = res_ln_outputs[1];

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
            false,                       /*apply_rotary_embedding*/
            false,                       /*scaling query*/
            1.0f,                        /*scaling factor*/
            true,                        /*qk_prod_scaling*/
            false,                       /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention")
                .c_str() /*name*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }

    ff.residual_layer_norm(
        hidden_states,
        mha,
        nullptr,
        res_ln_outputs,
        false,
        axes,
        true,
        startcoder_config.layer_norm_epsilon,
        true,
        DT_NONE,
        std::string("layers_" + std::to_string(i) + "_ln_2").c_str());
    residual = res_ln_outputs[0];
    Tensor l2_norm = res_ln_outputs[1];

    // mlp
    Tensor c_fc = ff.dense(
        l2_norm,
        startcoder_config.intermediate_size,
        AC_MODE_NONE,
        true,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers_" + std::to_string(i) + "_mlp_c_fc").c_str());

    c_fc = ff.gelu(c_fc);

    c_proj = ff.dense(
        c_fc,
        startcoder_config.hidden_size,
        AC_MODE_NONE,
        true,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers_" + std::to_string(i) + "_mlp_c_proj").c_str());
  }
  // final normalization and linear
  ff.residual_layer_norm(residual,
                         c_proj,
                         nullptr,
                         res_ln_outputs,
                         false,
                         axes,
                         true,
                         startcoder_config.layer_norm_epsilon,
                         true,
                         DT_NONE,
                         "transformer_ln_f");
  Tensor ln_f = res_ln_outputs[1];

  Tensor lm_head = ff.dense(ln_f,
                            startcoder_config.vocab_size,
                            AC_MODE_NONE,
                            false,
                            DT_NONE,
                            nullptr,
                            nullptr,
                            nullptr,
                            REG_MODE_NONE,
                            0.0f,
                            "lm_head");

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(lm_head, -1);
    // output = ff.beam_top_k(softmax, startcoder_config.max_beam_width, false);
    output = ff.argmax(softmax, /*beam_Search*/ true);
  } else {
    // Tensor softmax = ff.softmax(dense, -1);
    if (generationConfig.do_sample) {
      lm_head = ff.scalar_truediv(lm_head, generationConfig.temperature, false);
      Tensor softmax = ff.softmax(lm_head, -1);
      output = ff.sampling(softmax, generationConfig.topp);
    } else {
      // output = ff.arg_top_k(lm_head, /*k=*/1, false);
      output = ff.argmax(lm_head, /*beam_Search*/ false);
    }
  }

  InferenceManager *im = InferenceManager::get_inference_manager();
  FileDataLoader *fileloader = new FileDataLoader(
      "",
      weight_file_path,
      startcoder_config.num_attention_heads,
      1,
      startcoder_config.hidden_size,
      startcoder_config.hidden_size / startcoder_config.num_attention_heads,
      ff.config.tensor_parallelism_degree,
      use_full_precision);
  im->register_model_weights_loader(&ff, fileloader);
}

}; // namespace FlexFlow
