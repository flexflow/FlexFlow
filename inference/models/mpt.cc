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

#include "mpt.h"

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void MPT::create_mpt_model(FFModel &ff,
                           std::string const &model_config_file_path,
                           std::string const &weight_file_path,
                           InferenceMode mode,
                           GenerationConfig generationConfig,
                           bool use_full_precision) {
  MPTConfig mpt_config(model_config_file_path);
  mpt_config.print();

  if (ff.config.tensor_parallelism_degree > mpt_config.n_heads ||
      mpt_config.n_heads % ff.config.tensor_parallelism_degree != 0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;

  //------------------------------ build the model --------------------------
  Tensor input;
  {
    int const token_dims[] = {
        (mode == TREE_VERIFY_MODE || mode == BEAM_SEARCH_MODE)
            ? BatchConfig::max_verify_tokens_per_batch()
            : BatchConfig::max_tokens_per_batch(),
        1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  std::vector<int> axes = {0};

  Tensor hidden_states = ff.embedding(input,
                                      mpt_config.vocab_size,
                                      mpt_config.hidden_size,
                                      AGGR_MODE_NONE,
                                      use_full_precision ? DT_FLOAT : DT_HALF,
                                      NULL,
                                      embed_init,
                                      "transformer_wte");

  Tensor intermediate_output = nullptr, layernorm_output = nullptr;
  Tensor res_ln_outputs[2] = {nullptr, nullptr};

  for (int i = 0; i < mpt_config.n_layers; i++) {
    ff.set_transformer_layer_id(i);

    if (i == 0) {
      layernorm_output = ff.layer_norm(
          hidden_states,
          axes,
          true,
          1e-05,
          false,
          DT_NONE,
          std::string("layers_" + std::to_string(i) + "_norm_1").c_str());
    } else {
      ff.residual_layer_norm(
          intermediate_output,
          hidden_states,
          nullptr,
          res_ln_outputs,
          false,
          axes,
          true,
          1e-05,
          false,
          DT_NONE,
          std::string("layers_" + std::to_string(i) + "_norm_1").c_str());
      hidden_states = res_ln_outputs[0];
      layernorm_output = res_ln_outputs[1];
    }

    Tensor attn_outputs;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        attn_outputs = ff.spec_inc_multihead_self_attention(
            layernorm_output,
            mpt_config.hidden_size,
            mpt_config.n_heads,
            mpt_config.hidden_size / mpt_config.n_heads,
            mpt_config.hidden_size / mpt_config.n_heads,
            0.0f,
            false,
            false,
            false,
            DT_NONE, /*data_type*/
            NULL,
            false,
            /*scaling query*/ true,
            /*scaling factor*/
            pow((mpt_config.hidden_size / mpt_config.n_heads), -0.5),
            /*qk_prod_scaling*/ false,
            /*position_bias*/ true,
            std::string("layers_" + std::to_string(i) + "_attention")
                .c_str() /*name*/
        );
        break;
      }
      case TREE_VERIFY_MODE: {
        attn_outputs = ff.inc_multihead_self_attention_verify(
            layernorm_output,
            mpt_config.hidden_size,
            mpt_config.n_heads,
            mpt_config.hidden_size / mpt_config.n_heads,
            mpt_config.hidden_size / mpt_config.n_heads,
            0.0f,
            false,
            false,
            false,
            DT_NONE, /*data_type*/
            NULL,
            false,
            /*scaling query*/ true,
            /*scaling factor*/
            pow((mpt_config.hidden_size / mpt_config.n_heads), -0.5),
            /*qk_prod_scaling*/ false,
            /*position_bias*/ true,
            std::string("layers_" + std::to_string(i) + "_attention")
                .c_str() /*name*/
        );
        break;
      }
      case INC_DECODING_MODE: {
        attn_outputs = ff.inc_multihead_self_attention(
            layernorm_output,
            mpt_config.hidden_size,
            mpt_config.n_heads,
            mpt_config.hidden_size / mpt_config.n_heads,
            mpt_config.hidden_size / mpt_config.n_heads,
            0.0f,
            false,
            false,
            false,
            DT_NONE, /*data_type*/
            NULL,
            false,
            /*scaling query*/ true,
            /*scaling factor*/
            pow((mpt_config.hidden_size / mpt_config.n_heads), -0.5),
            /*qk_prod_scaling*/ false,
            /*position_bias*/ true,
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
        attn_outputs,
        hidden_states,
        nullptr,
        res_ln_outputs,
        false,
        axes,
        true,
        1e-05,
        false,
        DT_NONE,
        std::string("layers_" + std::to_string(i) + "_norm_2").c_str());
    hidden_states = res_ln_outputs[0];
    layernorm_output = res_ln_outputs[1];

    // MLP
    layernorm_output = ff.dense(
        layernorm_output,
        4 * mpt_config.hidden_size,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers_" + std::to_string(i) + "_ffn_up_proj").c_str());
    layernorm_output = ff.gelu(layernorm_output);
    intermediate_output = ff.dense(
        layernorm_output,
        mpt_config.hidden_size,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers_" + std::to_string(i) + "_ffn_down_proj").c_str());
  }

  // final
  ff.residual_layer_norm(intermediate_output,
                         hidden_states,
                         nullptr,
                         res_ln_outputs,
                         false,
                         axes,
                         true,
                         1e-05,
                         false,
                         DT_NONE,
                         "transformer_norm_f");
  Tensor all_final_norm = res_ln_outputs[1];

  Tensor lm_head = ff.dense(all_final_norm,
                            mpt_config.vocab_size,
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
    output = ff.argmax(softmax, /*beam_Search*/ true);
  } else {
    output = ff.argmax(lm_head, /*beam_Search*/ false);
  }
  FileDataLoader *fileloader =
      new FileDataLoader("",
                         weight_file_path,
                         mpt_config.n_heads,
                         mpt_config.n_heads,
                         mpt_config.hidden_size,
                         mpt_config.hidden_size / mpt_config.n_heads,
                         ff.config.tensor_parallelism_degree,
                         use_full_precision);

  InferenceManager *im = InferenceManager::get_inference_manager();
  im->register_model_weights_loader(&ff, fileloader);
}

}; // namespace FlexFlow
