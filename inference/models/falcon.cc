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
using json = nlohmann::json;

void FALCON::create_falcon_model(FFModel &ff,
                                 std::string const &model_config_file_path,
                                 std::string const &weight_file_path,
                                 InferenceMode mode,
                                 bool use_full_precision) {
  FalconConfig falcon_config(model_config_file_path);
  falcon_config.print();

  if (ff.config.tensor_parallelism_degree > falcon_config.n_head ||
      falcon_config.n_head % ff.config.tensor_parallelism_degree != 0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;

  Tensor input;
  {
    // assert(falcon_config.max_num_tokens <= BatchConfig::MAX_NUM_TOKENS);
    int const token_dims[] = {
        (mode == TREE_VERIFY_MODE || mode == BEAM_SEARCH_MODE)
            ? BatchConfig::max_verify_tokens_per_batch()
            : BatchConfig::max_tokens_per_batch(),
        1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  std::vector<int> axes = {0};

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  Tensor token = ff.embedding(input,
                              falcon_config.vocab_size,
                              falcon_config.hidden_size,
                              AGGR_MODE_NONE,
                              use_full_precision ? DT_FLOAT : DT_HALF,
                              NULL,
                              embed_init,
                              "word_embeddings");

  Tensor mha = nullptr, mlp_output = nullptr;
  Tensor qkv_proj = nullptr, o_proj = nullptr;
  Tensor res_ln_outputs[2] = {nullptr, nullptr};

  for (int i = 0; i < falcon_config.n_layer; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);

    // step 1: attention
    Tensor att_norm = nullptr;
    if (i == 0) {
      att_norm = ff.layer_norm(
          token,
          axes,
          true,
          falcon_config.layer_norm_epsilon,
          true,
          DT_NONE,
          std::string("layers." + std::to_string(i) + ".input_layernorm")
              .c_str());
    } else {
      ff.residual_layer_norm(
          token,
          mha,
          mlp_output,
          res_ln_outputs,
          true,
          axes,
          true,
          falcon_config.layer_norm_epsilon,
          true,
          false,
          DT_NONE,
          std::string("layers." + std::to_string(i) + ".input_layernorm")
              .c_str());
      token = res_ln_outputs[0];
      att_norm = res_ln_outputs[1];
    }

    qkv_proj = ff.dense(
        att_norm,
        falcon_config.hidden_size *
            3, // q, k, v. need to change if want to remove replication.
               // (q_heads + 2 * kv_heads) * proj_size
        AC_MODE_NONE,
        false,         // seems like it does not use bias
        DT_NONE,       // what is this
        nullptr,       // ?
        nullptr,       // ?
        nullptr,       // ?
        REG_MODE_NONE, // no regularization
        0.0f,          // no dropout
        std::string("layers." + std::to_string(i) + ".self_attention.qkv_proj")
            .c_str());
    qkv_proj->print("qkv_proj");

    switch (mode) {
      case BEAM_SEARCH_MODE: {
        o_proj = ff.spec_inc_multiquery_self_attention(
            qkv_proj,
            falcon_config.hidden_size,
            falcon_config.n_head,
            falcon_config.n_head_kv,
            falcon_config.hidden_size / falcon_config.n_head,
            falcon_config.hidden_size / falcon_config.n_head,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            NULL,    /*kernel_initializer*/
            falcon_config.rotary_embedding_meta,
            false, /*scaling query*/
            1.0f,  /*scaling factor*/
            true,  /*qk_prod_scaling*/
            false, /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attention")
                .c_str() /*name*/
        );
        break;
      }

      case TREE_VERIFY_MODE: {
        o_proj = ff.inc_multiquery_self_attention_verify(
            qkv_proj,
            falcon_config.hidden_size,
            falcon_config.n_head,
            falcon_config.n_head_kv,
            falcon_config.hidden_size / falcon_config.n_head,
            falcon_config.hidden_size / falcon_config.n_head,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            falcon_config.rotary_embedding_meta,
            false, /*scaling query*/
            1.0f,  /*scaling factor*/
            true,  /*qk_prod_scaling*/
            false, /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attention")
                .c_str() /*name*/
        );
        break;
      }

      case INC_DECODING_MODE: {
        o_proj = ff.inc_multiquery_self_attention(
            qkv_proj,
            falcon_config.hidden_size,
            falcon_config.n_head,
            falcon_config.n_head_kv,
            falcon_config.hidden_size / falcon_config.n_head,
            falcon_config.hidden_size / falcon_config.n_head,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            falcon_config.rotary_embedding_meta,
            false, /*scaling query*/
            1.0f,  /*scaling factor*/
            true,  /*qk_prod_scaling*/
            false, /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attention")
                .c_str() /*name*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }

    mha = ff.dense(
        o_proj,
        falcon_config.hidden_size,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers." + std::to_string(i) + ".self_attention.o_proj")
            .c_str());
    mha->print("mha");

    Tensor dense_h_to_4h = ff.dense(
        att_norm,
        falcon_config.hidden_size * 4,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers." + std::to_string(i) + ".mlp.dense_h_to_4h")
            .c_str());

    dense_h_to_4h = ff.gelu(dense_h_to_4h);

    mlp_output = ff.dense(
        dense_h_to_4h,
        falcon_config.hidden_size,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers." + std::to_string(i) + ".mlp.dense_4h_to_h")
            .c_str());
  }
  // final normalization and linear
  ff.residual_layer_norm(token,
                         mha,
                         mlp_output,
                         res_ln_outputs,
                         true,
                         axes,
                         true,
                         falcon_config.layer_norm_epsilon,
                         true,
                         false,
                         DT_NONE,
                         "ln_f");
  Tensor ln_f = res_ln_outputs[1];

  Tensor lm_head = ff.dense(ln_f,
                            falcon_config.vocab_size,
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
                         falcon_config.n_head,
                         falcon_config.n_head_kv,
                         falcon_config.hidden_size,
                         falcon_config.hidden_size / falcon_config.n_head,
                         ff.config.tensor_parallelism_degree,
                         use_full_precision);

  InferenceManager *im = InferenceManager::get_inference_manager();
  im->register_model_weights_loader(&ff, fileloader);
}

}; // namespace FlexFlow
