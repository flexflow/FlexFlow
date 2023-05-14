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
    int n_heads, n_layers, vocab_size, dim, multiple_of, hidden_dim,
        total_requests, incremental_mode, max_seq_len, max_num_tokens,
        max_beam_width, max_beam_depth;
    float norm_eps;
    std::string weight_file_path;
    std::string input_path;
    std::string tokenizer_file_path;
  };

  struct Small_Config : public Config {
    Small_Config(void) {
      vocab_size = 50265;
      dim = 768;
      n_heads = 12;
      n_layers = 12;
      hidden_dim = 3072;
    }
  };

  static void create_llama_model(FFModel &ff,
                                 InferenceManager &im,
                                 Config const &llama_config,
                                 int num_pipeline_stages,
                                 InferenceMode mode);
};

}; // namespace FlexFlow
