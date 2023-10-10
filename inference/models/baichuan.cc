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

#include "baichuan.h"       

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void BAICHUAN::create_baichuan_model(FFModel &ff,
                                     std::string const &model_config_file_path,
                                     std::string const &weight_file_path,
                                     InferenceMode mode,
                                     GenerationConfig generationConfig,
                                     bool use_full_precision ) {
   BAICHUANConfig baichuan_config(model_config_file_path);
   baichuan_config.print();

  if (ff.config.tensor_parallelism_degree > baichuan_config.num_attention_heads ||
      baichuan_config.num_attention_heads % ff.config.tensor_parallelism_degree !=
          0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;   

   Tensor input;
  {
    int const token_dims[] = {BatchConfig::max_tokens_per_batch(), 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token;

 //Note: the embedding layer may change 
  if (use_full_precision) {
    token = ff.embedding(input,
                         baichuan_config.vocab_size,
                         baichuan_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_FLOAT,
                         NULL,
                         embed_init,
                         "token_embedding");
  } else {
    token = ff.embedding(input,
                         baichuan_config.vocab_size,
                         baichuan_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_HALF,
                         NULL,
                         embed_init,
                         "token_embedding");
  }

  Layer *embedding = ff.layers.back();
  weights_layers.emplace("token_embedding_weight", embedding);
  Tensor w2 = nullptr;
  std::cout<<"baichuan_config.num_hidden_layers:"<<baichuan_config.num_hidden_layers<<" and use_full_precision:"<<use_full_precision <<std::endl;
  for(int i = 0; i < baichuan_config.num_hidden_layers; i++) {
    //set transformer layer id
    ff.set_transformer_layer_id(i);
    Tensor rms_norm = nullptr;
    Tensor token_rms_norm[2] = {nullptr, nullptr};
    if(i == 0) {
      rms_norm = ff.rms_norm(
        token,
        baichuan_config.rms_norm_eps,
        baichuan_config.hidden_size,
        DT_NONE, 
        std::string("layers_" + std::to_string(i) + "_attention_norm").c_str());
    } else {
      ff.residual_rms_norm(
        token,
        w2,
        token_rms_norm,
        baichuan_config.rms_norm_eps,
        baichuan_config.hidden_size,
        DT_NONE, 
        std::string("layers_" + std::to_string(i) + "_attention_norm").c_str());
        token = token_rms_norm[0];
        rms_norm = token_rms_norm[1];
    }


    //step 2: self attention
    Tensor mha;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multihead_self_attention(
            rms_norm,
            baichuan_config.hidden_size,
            baichuan_config.num_attention_heads,
            baichuan_config.hidden_size / baichuan_config.num_attention_heads,
            baichuan_config.hidden_size / baichuan_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*qkv_bias*/
            false,   /*final_bias*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr,    /*kernel_initializer*/
            true,    /*apply_rotary_embedding*/
            false,   /*scaling query*/
            1.0f,    /*scaling factor*/
            true,    /*qk_prod_scaling*/
            false,   /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention").c_str());
        break;
      }
      case TREE_VERIFY_MODE: {
        mha = ff.inc_multihead_self_attention_verify(
            rms_norm,
            baichuan_config.hidden_size,
            baichuan_config.num_attention_heads,
            baichuan_config.hidden_size / baichuan_config.num_attention_heads,
            baichuan_config.hidden_size / baichuan_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true,     /*apply_rotary_embedding*/
            false,   /*scaling query*/
            1.0f,    /*scaling factor*/
            true,    /*qk_prod_scaling*/
            false,   /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention").c_str());
        break;
      }
      case INC_DECODING_MODE: {
        std::cout<<"ff.inc_multihead_self_attention"<<std::endl;
        mha = ff.inc_multihead_self_attention(
            rms_norm,
            baichuan_config.hidden_size,
            baichuan_config.num_attention_heads,
            baichuan_config.hidden_size / baichuan_config.num_attention_heads,
            baichuan_config.hidden_size / baichuan_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true ,    /*apply_rotary_embedding*/
            false,   /*scaling query*/
            1.0f,    /*scaling factor*/
            true,    /*qk_prod_scaling*/
            false,   /*position_bias*/
            std::string("layers_" + std::to_string(i) + "_attention").c_str());
        break;
      }
      default: {
        assert(false);
      }
    }

    Layer *attention_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_attention_weight",
                           attention_layer);

    std::string layer_name = "layers_" + std::to_string(i) + "_ffn_norm";

    Tensor token_ff_norm[2] = {nullptr, nullptr};
    ff.residual_rms_norm(token,
                         mha,
                         token_ff_norm,
                         baichuan_config.rms_norm_eps,
                         baichuan_config.hidden_size,
                         DT_NONE,
                         layer_name.c_str());

    token = token_ff_norm[0];
    Tensor ff_norm = token_ff_norm[1];

    layer_name = "layers_" + std::to_string(i) + "_feed_forward_w1";
    Tensor w1 = ff.dense(ff_norm,
                         baichuan_config.intermediate_size,
                         AC_MODE_NONE,
                         false,
                         DT_NONE,
                         nullptr,
                         nullptr,
                         nullptr,
                         REG_MODE_NONE,
                         0.0f,
                         layer_name.c_str());

    layer_name = "layers_" + std::to_string(i) + "_feed_forward_w3";
    Tensor w3 = ff.dense(ff_norm,
                         baichuan_config.intermediate_size,
                         AC_MODE_NONE,
                         false,
                         DT_NONE,
                         nullptr,
                         nullptr,
                         nullptr,
                         REG_MODE_NONE,
                         0.0f,
                         layer_name.c_str());

    Tensor multi = ff.sigmoid_silu_multi(w1, w3);

    layer_name = "layers_" + std::to_string(i) + "_feed_forward_w2";
    w2 = ff.dense(multi,
                         baichuan_config.hidden_size,
                         AC_MODE_NONE,
                         false,
                         DT_NONE,
                         nullptr,
                         nullptr,
                         nullptr,
                         REG_MODE_NONE,
                         0.0f,
                         layer_name.c_str());

    token = ff.add(token, w2);
  }

  // final normalization and linear
  Tensor final_rms_norm_output[2] = {nullptr, nullptr};
  ff.residual_rms_norm(token,
                       w2,
                       final_rms_norm_output,
                       baichuan_config.rms_norm_eps,
                       baichuan_config.hidden_size,
                       DT_NONE,
                       "norm");

  Tensor dense = ff.dense(final_rms_norm_output[1],
                          baichuan_config.vocab_size,
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
    // output = ff.beam_top_k(softmax, baichuan_config.max_beam_width, false);
    output = ff.argmax(softmax, /*beam_Search*/ true);
  } else {
    // Tensor softmax = ff.softmax(dense, -1);
    if (generationConfig.do_sample) {
      dense = ff.scalar_truediv(dense, generationConfig.temperature, false);
      Tensor softmax = ff.softmax(dense, -1);
      output = ff.sampling(softmax, generationConfig.topp);
    } else {
      // output = ff.arg_top_k(dense, /*k=*/1, false);
      output = ff.argmax(dense, /*beam_Search*/ false);
    }
  }

    // Compile the model
  std::cout << "------start compile ----------" << std::endl;
  InferenceManager *im = InferenceManager::get_inference_manager();
  im->compile_model_and_allocate_buffer(&ff);
  FileDataLoader fileloader("",
                            weight_file_path,
                            baichuan_config.num_attention_heads,
                            baichuan_config.num_attention_heads,
                            baichuan_config.hidden_size,
                            baichuan_config.hidden_size /
                                baichuan_config.num_attention_heads,
                            ff.config.tensor_parallelism_degree);
  fileloader.load_weights(&ff, use_full_precision);
  std::cout << "------load weight finished----------" << std::endl;

  // init operators
  im->init_operators_inference(&ff);

}

} // namespace FlexFlow