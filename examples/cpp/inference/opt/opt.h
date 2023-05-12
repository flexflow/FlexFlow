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
#include "inference_config.h"
struct OptConfig : InferenceConfig {
  OptConfig(void) : InferenceConfig() {
    vocab_size = 50272, word_embed_proj_dim = 768, hidden_size = 768;
    max_position_embeddings = 2048;
    layer_norm_elementwise_affine = true;
    num_attention_heads = 12;
    dropout = 0.1;
    seed = 3;
    ffn_dim = 3072;
    num_hidden_layers = 12;
  }
  int word_embed_proj_dim;
  std::string input_path;
  std::string weight_file_path;
  int max_position_embeddings;
  bool layer_norm_elementwise_affine;
  float dropout;
  unsigned long long seed;
  int ffn_dim;
  int num_hidden_layers;
};
