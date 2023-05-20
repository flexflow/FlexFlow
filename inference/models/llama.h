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

class LLAMA {
public:
  struct Config {
    Config(void) {
      // todo read from config/param file
      n_layers = 32;
      vocab_size = 32000;
      n_heads = 32;
      dim = 4096;
      multiple_of = 256;
      norm_eps = 1e-6;
      total_requests = 2560;
      incremental_mode = true;
      max_seq_len = BatchConfig::MAX_SEQ_LENGTH;
      max_num_tokens = BatchConfig::MAX_NUM_TOKENS;
      max_beam_width = BeamSearchBatchConfig::MAX_BEAM_WIDTH;
      max_beam_depth = BeamSearchBatchConfig::MAX_BEAM_DEPTH;

      // hidden dim
      hidden_dim = 4 * dim;
      hidden_dim = int(2 * hidden_dim / 3);
      hidden_dim =
          multiple_of * int((hidden_dim + multiple_of - 1) / multiple_of);
    }

    Config(std::string config_filepath) {
      std::ifstream config_file(config_filepath);
      if (config_file.is_open()) {
        try {
          json config_json;
          config_file >> config_json;

          n_layers = config_json["n_layers"];
          vocab_size = config_json["vocab_size"];
          n_heads = config_json["n_heads"];
          dim = config_json["dim"];
          multiple_of = config_json["multiple_of"];
          norm_eps = config_json["norm_eps"];
          total_requests = config_json["total_requests"];
          incremental_mode = config_json["incremental_mode"];
          max_seq_len = config_json["max_seq_len"];
          max_num_tokens = config_json["max_num_tokens"];
          max_beam_width = config_json["max_beam_width"];
          max_beam_depth = config_json["max_beam_depth"];
          hidden_dim = config_json["hidden_dim"];
          weight_file_path = config_json["weight_file_path"];
          input_path = config_json["input_path"];
          tokenizer_file_path = config_json["tokenizer_file_path"];
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
      std::cout << "n_layers: " << n_layers << std::endl;
      std::cout << "vocab_size: " << vocab_size << std::endl;
      std::cout << "n_heads: " << n_heads << std::endl;
      std::cout << "dim: " << dim << std::endl;
      std::cout << "multiple_of: " << multiple_of << std::endl;
      std::cout << "norm_eps: " << norm_eps << std::endl;
      std::cout << "total_requests: " << total_requests << std::endl;
      std::cout << "incremental_mode: " << incremental_mode << std::endl;
      std::cout << "max_seq_len: " << max_seq_len << std::endl;
      std::cout << "max_num_tokens: " << max_num_tokens << std::endl;
      std::cout << "max_beam_width: " << max_beam_width << std::endl;
      std::cout << "max_beam_depth: " << max_beam_depth << std::endl;
      std::cout << "hidden_dim: " << hidden_dim << std::endl;
      std::cout << "weight_file_path: " << weight_file_path << std::endl;
      std::cout << "input_path: " << input_path << std::endl;
      std::cout << "tokenizer_file_path: " << tokenizer_file_path << std::endl;
    }

    int n_heads, n_layers, vocab_size, dim, multiple_of, hidden_dim,
        total_requests, incremental_mode, max_seq_len, max_num_tokens,
        max_beam_width, max_beam_depth;
    float norm_eps;
    std::string weight_file_path;
    std::string input_path;
    std::string tokenizer_file_path;
  };

  static void create_llama_model(FFModel &ff,
                                 InferenceManager &im,
                                 std::string const &model_config_file_path,
                                 std::string const &weight_file_path,
                                 int num_pipeline_stages,
                                 InferenceMode mode);
};

}; // namespace FlexFlow
