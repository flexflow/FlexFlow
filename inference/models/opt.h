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

class OPT {
public:
  struct Config {
    Config(void) {
      vocab_size = 50272;
      word_embed_proj_dim = 4096;
      hidden_size = 4096;
      max_position_embeddings = 2048;
      layer_norm_elementwise_affine = true;
      num_hidden_layers = 32;
      dropout = 0.1;
      ffn_dim = 16384;
      max_beam_width = 1;
      batchSize = 8;
      sentence_len = 100;
      max_beam_depth = 4;
    }
    int vocab_size;
    int word_embed_proj_dim;
    int hidden_size;
    int num_attention_heads;
    std::string input_path;
    std::string weight_file_path;
    std::string tokenizer_assets_folder;
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

  struct Small_Config : public Config {
    Small_Config(void) {
      word_embed_proj_dim = 768;
      hidden_size = 768;
      num_attention_heads = 12;
      ffn_dim = 3072;
      num_hidden_layers = 12;
    }
  };

  static void create_opt_model(FFModel &ff,
                               InferenceManager &im,
                               Config const &opt_config,
                               int num_pipeline_stages,
                               InferenceMode mode);
};

}; // namespace FlexFlow
