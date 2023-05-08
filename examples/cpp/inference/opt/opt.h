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

#include "inference_config.h"
#include "file_loader.h"
// # OPTConfig {
// #   "_remove_final_layer_norm": false,
// #   "activation_function": "relu",
// #   "attention_dropout": 0.0,
// #   "bos_token_id": 2,
// #   "do_layer_norm_before": true,
// #   "dropout": 0.1,
// #   "enable_bias": true,
// #   "eos_token_id": 2,
// #   "ffn_dim": 3072,
// #   "hidden_size": 768,
// #   "init_std": 0.02,
// #   "layer_norm_elementwise_affine": true,
// #   "layerdrop": 0.0,
// #   "max_position_embeddings": 2048,
// #   "model_type": "opt",
// #   "num_attention_heads": 12,
// #   "num_hidden_layers": 12,
// #   "pad_token_id": 1,
// #   "transformers_version": "4.27.2",
// #   "use_cache": true,
// #   "vocab_size": 50272,
// #   "word_embed_proj_dim": 768
// # }
struct OptConfig : InferenceConfig {
  OptConfig(void) : InferenceConfig() {
    vocab_size = 50272,
    word_embed_proj_dim = 768,
    hidden_size = 768;
    max_position_embeddings = 2048;
    layer_norm_elementwise_affine = true;
    weight_file_path = "/home/ubuntu/FlexFlow/examples/cpp/inference/opt/weights/";
  }
  int word_embed_proj_dim;
  std::string input_path;
  std::string weight_file_path;
  int max_position_embeddings;
  bool layer_norm_elementwise_affine;
};
