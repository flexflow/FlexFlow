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

#include "llama.h"

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void LLAMA::create_llama_model(FFModel &ff,
                               std::string const &model_config_file_path,
                               std::string const &weight_file_path,
                               InferenceMode mode,
                               GenerationConfig generation_config,
                               bool use_full_precision) {
  // do not apply cpu offload in beam search model.
  LLAMAConfig llama_config(model_config_file_path);
  llama_config.print();

  if (ff.config.tensor_parallelism_degree > llama_config.num_attention_heads ||
      llama_config.num_attention_heads % ff.config.tensor_parallelism_degree !=
          0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;

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

  Tensor token = ff.embedding(input,
                              llama_config.vocab_size,
                              llama_config.hidden_size,
                              AGGR_MODE_NONE,
                              use_full_precision ? DT_FLOAT : DT_HALF,
                              NULL,
                              embed_init,
                              "tok_embeddings");

  Tensor w2 = nullptr;

  for (int i = 0; i < llama_config.num_hidden_layers; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);

    // step 1: attention
    Tensor att_norm = nullptr;
    Tensor token_att_norm[2] = {nullptr, nullptr};
    if (i == 0) {
      att_norm = ff.rms_norm(
          token,
          llama_config.rms_norm_eps,
          llama_config.hidden_size,
          DT_NONE,
          std::string("layers_" + std::to_string(i) + "_attention_norm")
              .c_str());
    } else {
      ff.residual_rms_norm(
          token,
          w2,
          token_att_norm,
          llama_config.rms_norm_eps,
          llama_config.hidden_size,
          DT_NONE,
          std::string("layers_" + std::to_string(i) + "_attention_norm")
              .c_str());
      token = token_att_norm[0];
      att_norm = token_att_norm[1];
    }

    Tensor mha;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multihead_self_attention(
            att_norm,
            llama_config.hidden_size,
            llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*qkv_bias*/
            false,   /*final_bias*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            NULL,    /*kernel_initializer*/
            true,    /*apply_rotary_embedding*/
            false,   /*scaling query*/
            1.0f,    /*scaling factor*/
            true,    /*qk_prod_scaling*/
            false,   /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention")
                .c_str() /*name*/
        );
        break;
      }
      case TREE_VERIFY_MODE: {
        mha = ff.inc_multihead_self_attention_verify(
            att_norm,
            llama_config.hidden_size,
            llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*qkv_bias*/
            false,   /*final_bias*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true,    /*apply_rotary_embedding*/
            false,   /*scaling query*/
            1.0f,    /*scaling factor*/
            true,    /*qk_prod_scaling*/
            false,   /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention")
                .c_str() /*name*/
        );
        break;
      }
      case INC_DECODING_MODE: {
        mha = ff.inc_multihead_self_attention(
            att_norm,
            llama_config.hidden_size,
            llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*qkv_bias*/
            false,   /*final_bias*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true,    /*apply_rotary_embedding*/
            false,   /*scaling query*/
            1.0f,    /*scaling factor*/
            true,    /*qk_prod_scaling*/
            false,   /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention")
                .c_str() /*name*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }

    // step 2: SILU activaion
    Tensor token_ff_norm[2] = {nullptr, nullptr};
    ff.residual_rms_norm(
        token,
        mha,
        token_ff_norm,
        llama_config.rms_norm_eps,
        llama_config.hidden_size,
        DT_NONE,
        std::string("layers_" + std::to_string(i) + "_ffn_norm").c_str());
    token = token_ff_norm[0];
    Tensor ff_norm = token_ff_norm[1];

    Tensor w1 =
        ff.dense(ff_norm,
                 llama_config.intermediate_size,
                 AC_MODE_NONE,
                 false,
                 DT_NONE,
                 nullptr,
                 nullptr,
                 nullptr,
                 REG_MODE_NONE,
                 0.0f,
                 std::string("layers_" + std::to_string(i) + "_feed_forward_w1")
                     .c_str());

    Tensor w3 =
        ff.dense(ff_norm,
                 llama_config.intermediate_size,
                 AC_MODE_NONE,
                 false,
                 DT_NONE,
                 nullptr,
                 nullptr,
                 nullptr,
                 REG_MODE_NONE,
                 0.0f,
                 std::string("layers_" + std::to_string(i) + "_feed_forward_w3")
                     .c_str());

    Tensor multi = ff.sigmoid_silu_multi(w1, w3);

    w2 =
        ff.dense(multi,
                 llama_config.hidden_size,
                 AC_MODE_NONE,
                 false,
                 DT_NONE,
                 nullptr,
                 nullptr,
                 nullptr,
                 REG_MODE_NONE,
                 0.0f,
                 std::string("layers_" + std::to_string(i) + "_feed_forward_w2")
                     .c_str());
  }
  // final normalization and linear
  Tensor final_rms_norm_output[2] = {nullptr, nullptr};
  ff.residual_rms_norm(token,
                       w2,
                       final_rms_norm_output,
                       llama_config.rms_norm_eps,
                       llama_config.hidden_size,
                       DT_NONE,
                       "norm");

  Tensor dense = ff.dense(final_rms_norm_output[1],
                          llama_config.vocab_size,
                          AC_MODE_NONE,
                          false,
                          DT_NONE,
                          nullptr,
                          nullptr,
                          nullptr,
                          REG_MODE_NONE,
                          0.0f,
                          "output");

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(dense, -1);
    // output = ff.beam_top_k(softmax, llama_config.max_beam_width, false);
    // output = ff.argmax(softmax, /*beam_Search*/ true);
    output = ff.arg_top_k(softmax, llama_config.max_beam_width, false, true);
    // output = ff.top_k(softmax, )
  } else {
    // Tensor softmax = ff.softmax(dense, -1);
    if (generation_config.do_sample) {
      dense = ff.scalar_truediv(dense, generation_config.temperature, false);
      Tensor softmax = ff.softmax(dense, -1);
      output = ff.sampling(softmax, generation_config.topp);
    } else {
      // output = ff.arg_top_k(dense, /*k=*/1, false);
      output = ff.argmax(dense, /*beam_Search*/ false);
    }
  }

  FileDataLoader *fileloader = new FileDataLoader(
      "",
      weight_file_path,
      llama_config.num_attention_heads,
      llama_config.num_attention_heads,
      llama_config.hidden_size,
      llama_config.hidden_size / llama_config.num_attention_heads,
      ff.config.tensor_parallelism_degree,
      use_full_precision);

  InferenceManager *im = InferenceManager::get_inference_manager();
  im->register_model_weights_loader(&ff, fileloader);
}

}; // namespace FlexFlow
