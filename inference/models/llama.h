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

class LLAMA {
public:
  struct LLAMAConfig {
    LLAMAConfig(std::string const &model_config_file_path) {
      std::ifstream config_file(model_config_file_path);
      if (config_file.is_open()) {
        try {
          json model_config;
          config_file >> model_config;
          num_hidden_layers = model_config["num_hidden_layers"];
          vocab_size = model_config["vocab_size"];
          num_attention_heads = model_config["num_attention_heads"];
          hidden_size = model_config["hidden_size"];
          rms_norm_eps = model_config["rms_norm_eps"];
          intermediate_size = model_config["intermediate_size"];
        } catch (json::exception const &e) {
          std::cerr << "Error parsing LLAMA config from JSON file: " << e.what()
                    << std::endl;
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
      std::cout << "LLAMA Config:" << std::endl;
      std::cout << "\tnum_hidden_layers: " << num_hidden_layers << std::endl;
      std::cout << "\tvocab_size: " << vocab_size << std::endl;
      std::cout << "\tnum_attention_heads: " << num_attention_heads
                << std::endl;
      std::cout << "\thidden_size: " << hidden_size << std::endl;
      std::cout << "\trms_norm_eps: " << rms_norm_eps << std::endl;
      std::cout << "\tintermediate_size: " << intermediate_size << std::endl;

      // std::cout << "\tmax_seq_len: " << max_seq_len << std::endl;
      // std::cout << "\tmax_num_tokens: " << max_num_tokens << std::endl;
      std::cout << "\tmax_beam_width: " << max_beam_width << std::endl;
      std::cout << "\tmax_beam_depth: " << max_beam_depth << std::endl;
    }

    // int max_seq_len, max_num_tokens;
    int max_beam_width, max_beam_depth;
    int num_hidden_layers, vocab_size, num_attention_heads, hidden_size,
        intermediate_size;
    float rms_norm_eps;
  };

  static void create_llama_model(FFModel &ff,
                                 std::string const &model_config_file_path,
                                 std::string const &weight_file_path,
                                 InferenceMode mode,
                                 GenerationConfig generation_config,
                                 bool use_full_precision = false);
};

}; // namespace FlexFlow
