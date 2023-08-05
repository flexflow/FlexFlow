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
                               json model_config,
                               std::string const &weight_file_path,
                               InferenceMode mode,
                               SamplingConfig samplingConfig,
                               bool use_full_precision) {
  // do not apply cpu offload in beam search model.
  LLAMAConfig llama_config(model_config);
  llama_config.print();

  std::unordered_map<std::string, Layer *> weights_layers;

  Tensor input;
  {
    assert(llama_config.max_num_tokens <= BatchConfig::MAX_NUM_TOKENS);
    int const token_dims[] = {BatchConfig::MAX_NUM_TOKENS, 1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token;

  if (use_full_precision) {
    token = ff.embedding(input,
                         llama_config.vocab_size,
                         llama_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_FLOAT,
                         NULL,
                         embed_init);
  } else {
    token = ff.embedding(input,
                         llama_config.vocab_size,
                         llama_config.hidden_size,
                         AGGR_MODE_NONE,
                         DT_HALF,
                         NULL,
                         embed_init);
  }

  Layer *embedding = ff.layers.back();
  weights_layers.emplace("tok_embeddings_weight", embedding);

  for (int i = 0; i < llama_config.num_hidden_layers; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);
    // step 1: attention
    Tensor att_norm =
        ff.rms_norm(token, llama_config.rms_norm_eps, llama_config.hidden_size);
    Layer *attention_norm = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) +
                               "_attention_norm_weight",
                           attention_norm);

    Tensor mha;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multihead_self_attention(
            att_norm,
            llama_config.hidden_size,
            llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            llama_config.hidden_size / llama_config.num_attention_heads,
            0.0f,
            false,
            false,
            false,
            DT_NONE,
            NULL,
            true);
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
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true     /*apply_rotary_embedding*/
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
            false,   /*bias*/
            false,   /*add_bias_kv*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            true     /*apply_rotary_embedding*/
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
    token = ff.add(token, mha);

    // step 2: SILU activaion
    Tensor ff_norm =
        ff.rms_norm(token, llama_config.rms_norm_eps, llama_config.hidden_size);
    Layer *ffn_layer = ff.layers.back();
    weights_layers.emplace("layers_" + std::to_string(i) + "_ffn_norm_weight",
                           ffn_layer);

    Tensor w1 =
        ff.dense(ff_norm, llama_config.intermediate_size, AC_MODE_NONE, false);
    Layer *w1_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w1_weight", w1_layer);

    Tensor w3 =
        ff.dense(ff_norm, llama_config.intermediate_size, AC_MODE_NONE, false);
    Layer *w3_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w3_weight", w3_layer);

    Tensor sigmoid = ff.sigmoid(w1);
    Tensor silu = ff.multiply(w1, sigmoid);
    Tensor multi = ff.multiply(silu, w3);

    Tensor w2 = ff.dense(multi, llama_config.hidden_size, AC_MODE_NONE, false);
    Layer *w2_layer = ff.layers.back();
    weights_layers.emplace(
        "layers_" + std::to_string(i) + "_feed_forward_w2_weight", w2_layer);
    token = ff.add(token, w2);
  }
  // final normalization and linear
  std::vector<int> axes = {2};
  token =
      ff.rms_norm(token, llama_config.rms_norm_eps, llama_config.hidden_size);
  Layer *final_norm = ff.layers.back();
  weights_layers.emplace("norm_weight", final_norm);

  Tensor dense = ff.dense(token, llama_config.vocab_size, AC_MODE_NONE, false);
  Layer *final_linear = ff.layers.back();
  weights_layers.emplace("output_weight", final_linear);

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(dense, -1);
    // output = ff.beam_top_k(softmax, llama_config.max_beam_width, false);
    output = ff.argmax(softmax, /*beam_Search*/ true);
  } else {
    // Tensor softmax = ff.softmax(dense, -1);
    if (samplingConfig.do_sample) {
      dense = ff.scalar_truediv(dense, samplingConfig.temperature, false);
      Tensor softmax = ff.softmax(dense, -1);
      output = ff.sampling(softmax, samplingConfig.topp);
    } else {
      // output = ff.arg_top_k(dense, /*k=*/1, false);
      output = ff.argmax(dense, /*beam_Search*/ false);
    }
  }

  InferenceManager *im = InferenceManager::get_inference_manager();
  // Compile the model
  std::cout << "------start compile ----------" << std::endl;
  int tensor_partition_num = ff.config.tensor_parallelism_degree;
  im->compile_model_and_allocate_buffer(&ff);
  FileDataLoader fileloader("",
                            weight_file_path,
                            llama_config.num_attention_heads,
                            llama_config.num_attention_heads,
                            llama_config.hidden_size,
                            llama_config.hidden_size /
                                llama_config.num_attention_heads,
                            tensor_partition_num);
  fileloader.load_weights(&ff, weights_layers, use_full_precision);
  std::cout << "------load weight finished----------" << std::endl;

  // init operators
  im->init_operators_inference(&ff);
}

}; // namespace FlexFlow
