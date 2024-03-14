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
#pragma once

// #include "file_loader.h"
#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {

class OPT {
public:
  struct OPTConfig {
    OPTConfig(std::string const &model_config_file_path) {
      std::ifstream config_file(model_config_file_path);
      if (config_file.is_open()) {
        try {
          json model_config;
          config_file >> model_config;
          do_layer_norm_before = model_config["do_layer_norm_before"];
          dropout = model_config["dropout"];
          enable_bias = model_config["enable_bias"];
          ffn_dim = model_config["ffn_dim"];
          hidden_size = model_config["hidden_size"];
          layer_norm_elementwise_affine =
              model_config["layer_norm_elementwise_affine"];
          max_position_embeddings = model_config["max_position_embeddings"];
          num_attention_heads = model_config["num_attention_heads"];
          num_hidden_layers = model_config["num_hidden_layers"];
          vocab_size = model_config["vocab_size"];
          word_embed_proj_dim = model_config["word_embed_proj_dim"];
        } catch (json::exception const &e) {
          std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
          assert(false);
        }
      } else {
        std::cerr << "Error opening JSON file " << model_config_file_path
                  << std::endl;
        assert(false);
      }
      // max_seq_len = BatchConfig::MAX_SEQ_LENGTH;
      // max_num_tokens = BatchConfig::MAX_NUM_TOKENS;
      max_beam_width = BeamSearchBatchConfig::MAX_BEAM_WIDTH;
      max_beam_depth = BeamSearchBatchConfig::MAX_BEAM_DEPTH;
    }

    void print() const {
      std::cout << "OPT Config:" << std::endl;
      std::cout << "\tdo_layer_norm_before: " << do_layer_norm_before
                << std::endl;
      std::cout << "\tdropout: " << dropout << std::endl;
      std::cout << "\tenable_bias: " << enable_bias << std::endl;
      std::cout << "\tffn_dim: " << ffn_dim << std::endl;
      std::cout << "\thidden_size: " << hidden_size << std::endl;
      std::cout << "\tlayer_norm_elementwise_affine: "
                << layer_norm_elementwise_affine << std::endl;
      std::cout << "\tmax_position_embeddings: " << max_position_embeddings
                << std::endl;
      std::cout << "\tnum_attention_heads: " << num_attention_heads
                << std::endl;
      std::cout << "\tnum_hidden_layers: " << num_hidden_layers << std::endl;
      std::cout << "\tvocab_size: " << vocab_size << std::endl;
      std::cout << "\tword_embed_proj_dim: " << word_embed_proj_dim
                << std::endl;

      // std::cout << "\tmax_seq_len: " << max_seq_len << std::endl;
      // std::cout << "\tmax_num_tokens: " << max_num_tokens << std::endl;
      std::cout << "\tmax_beam_width: " << max_beam_width << std::endl;
      std::cout << "\tmax_beam_depth: " << max_beam_depth << std::endl;
    }

    // int max_seq_len, max_num_tokens;
    int max_beam_width, max_beam_depth;
    bool do_layer_norm_before, enable_bias, layer_norm_elementwise_affine;
    float dropout;
    int ffn_dim, hidden_size, max_position_embeddings, num_attention_heads,
        num_hidden_layers, vocab_size, word_embed_proj_dim;
  };

  static void create_opt_model(FFModel &ff,
                               std::string const &model_config_file_path,
                               std::string const &weight_file_path,
                               InferenceMode mode,
                               bool use_full_precision = false);
};

}; // namespace FlexFlow
