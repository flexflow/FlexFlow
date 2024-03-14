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

class FALCON {
public:
  struct FalconConfig {
    FalconConfig(std::string const &model_config_file_path) {
      std::ifstream config_file(model_config_file_path);
      if (config_file.is_open()) {
        try {
          json model_config;
          config_file >> model_config;
          bias = model_config["bias"];
          hidden_size = model_config["hidden_size"];
          layer_norm_epsilon = model_config["layer_norm_epsilon"];
          multi_query = model_config["multi_query"];
          n_head = (model_config.find("n_head") != model_config.end())
                       ? model_config["n_head"]
                       : model_config["num_attention_heads"];
          if (model_config.contains("n_head_kv")) {
            n_head_kv = model_config["n_head_kv"];
          } else {
            n_head_kv = 1;
          }
          n_layer = (model_config.find("n_layer") != model_config.end())
                        ? model_config["n_layer"]
                        : model_config["num_hidden_layers"];
          parallel_attn = model_config["parallel_attn"];
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
      std::cout << "Falcon Config:" << std::endl;
      std::cout << "\tbias: " << bias << std::endl;
      std::cout << "\thidden_size: " << hidden_size << std::endl;
      std::cout << "\tlayer_norm_epsilon: " << layer_norm_epsilon << std::endl;
      std::cout << "\tmulti_query: " << multi_query << std::endl;
      std::cout << "\tn_head: " << n_head << std::endl;
      std::cout << "\tn_head_kv: " << n_head << std::endl;
      std::cout << "\tn_layer: " << n_layer << std::endl;
      std::cout << "\tparallel_attn: " << parallel_attn << std::endl;
      std::cout << "\tvocab_size: " << vocab_size << std::endl;

      // std::cout << "\tmax_seq_len: " << max_seq_len << std::endl;
      // std::cout << "\tmax_num_tokens: " << max_num_tokens << std::endl;
      std::cout << "\tmax_beam_width: " << max_beam_width << std::endl;
      std::cout << "\tmax_beam_depth: " << max_beam_depth << std::endl;
    }

    bool bias, multi_query, parallel_attn;
    int hidden_size, n_head, n_head_kv, n_layer, vocab_size;
    float layer_norm_epsilon;
    // int max_seq_len, max_num_tokens;
    int max_beam_width, max_beam_depth;
  };

  static void create_falcon_model(FFModel &ff,
                                  std::string const &model_config_file_path,
                                  std::string const &weight_file_path,
                                  InferenceMode mode,
                                  bool use_full_precision = false);
};

}; // namespace FlexFlow
