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

#include "file_loader.h"
#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {

class OPT {
public:
  struct Config {
    Config(void) {
      vocab_size = 50272;
      word_embed_proj_dim = 4096;
      hidden_size = 4096;
      num_attention_heads = 32;
      max_position_embeddings = 2048;
      layer_norm_elementwise_affine = true;
      dropout = 0.1;
      ffn_dim = 16384;
      num_hidden_layers = 32;
      max_beam_width = 1;
      batchSize = 8;
      sentence_len = 100;
      max_beam_depth = 4;
    }
    Config(std::string config_filepath) {
      std::ifstream config_file(config_filepath);
      if (config_file.is_open()) {
        try {
          json config_json;
          config_file >> config_json;

          vocab_size = config_json["vocab_size"];
          word_embed_proj_dim = config_json["word_embed_proj_dim"];
          hidden_size = config_json["hidden_size"];
          num_attention_heads = config_json["num_attention_heads"];
          max_position_embeddings = config_json["max_position_embeddings"];
          layer_norm_elementwise_affine =
              config_json["layer_norm_elementwise_affine"];
          dropout = config_json["dropout"];
          ffn_dim = config_json["ffn_dim"];
          num_hidden_layers = config_json["num_hidden_layers"];
          max_beam_width = config_json["max_beam_width"];
          batchSize = config_json["batchSize"];
          sentence_len = config_json["sentence_len"];
          max_beam_depth = config_json["max_beam_depth"];
        } catch (json::exception const &e) {
          std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
          assert(false);
        }
      } else {
        std::cerr << "Error opening JSON file." << std::endl;
        assert(false);
      }
    }
    void printConfig() const {
      std::cout << "OPT Config:" << std::endl;
      std::cout << "vocab_size: " << vocab_size << std::endl;
      std::cout << "word_embed_proj_dim: " << word_embed_proj_dim << std::endl;
      std::cout << "hidden_size: " << hidden_size << std::endl;
      std::cout << "num_attention_heads: " << num_attention_heads << std::endl;
      std::cout << "max_position_embeddings: " << max_position_embeddings
                << std::endl;
      std::cout << "layer_norm_elementwise_affine: " << std::boolalpha
                << layer_norm_elementwise_affine << std::endl;
      std::cout << "dropout: " << dropout << std::endl;
      std::cout << "ffn_dim: " << ffn_dim << std::endl;
      std::cout << "num_hidden_layers: " << num_hidden_layers << std::endl;
      std::cout << "max_beam_width: " << max_beam_width << std::endl;
      std::cout << "batchSize: " << batchSize << std::endl;
      std::cout << "sentence_len: " << sentence_len << std::endl;
      std::cout << "max_beam_depth: " << max_beam_depth << std::endl;
    }
    int vocab_size;
    int word_embed_proj_dim;
    int hidden_size;
    int num_attention_heads;
    int max_position_embeddings;
    bool layer_norm_elementwise_affine;
    float dropout;
    int ffn_dim;
    int num_hidden_layers;
    int max_beam_width;
    int batchSize;
    int sentence_len;
    int max_beam_depth;
  };

  static void create_opt_model(FFModel &ff,
                               InferenceManager &im,
                               std::string const &model_config_file_path,
                               std::string const &weight_file_path,
                               int num_pipeline_stages,
                               InferenceMode mode,
                               bool use_full_precision = false);
};

}; // namespace FlexFlow
