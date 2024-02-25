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

class MPT {
public:
  struct MPTConfig {
    MPTConfig(std::string const &model_config_file_path) {
      std::ifstream config_file(model_config_file_path);
      if (config_file.is_open()) {
        try {
          json model_config;
          config_file >> model_config;
          hidden_size = model_config["d_model"];
          n_heads = model_config["n_heads"];
          n_layers = model_config["n_layers"];
          vocab_size = model_config["vocab_size"];
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
      std::cout << "MPT Config:" << std::endl;
      std::cout << "\thidden_size: " << hidden_size << std::endl;
      std::cout << "\tn_heads: " << n_heads << std::endl;
      std::cout << "\tn_layers: " << n_layers << std::endl;
      std::cout << "\tvocab_size: " << vocab_size << std::endl;
    }

    // int max_seq_len, max_num_tokens;
    int max_beam_width, max_beam_depth;
    int hidden_size, n_heads, n_layers, vocab_size;
  };

  static void create_mpt_model(FFModel &ff,
                               std::string const &model_config_file_path,
                               std::string const &weight_file_path,
                               InferenceMode mode,
                               GenerationConfig generationConfig,
                               bool use_full_precision = false);
};

}; // namespace FlexFlow
