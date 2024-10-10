/* Copyright 2022 CMU, Stanford, Facebook, LANL
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
#include "flexflow/batch_config.h"
#include <string>
#include <vector>

namespace FlexFlow {

struct GenerationConfig {
  bool do_sample = false;
  float temperature = 0.8;
  float topp = 0.6;
  GenerationConfig(bool _do_sample, float _temperature, float _topp) {
    temperature = _temperature > 0 ? _temperature : temperature;
    topp = _topp > 0 ? _topp : topp;
    do_sample = _do_sample;
  }
  GenerationConfig() {}
};

struct GenerationResult {
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;
  RequestGuid guid;
  std::string input_text;
  std::string output_text;
  std::vector<TokenId> input_tokens;
  std::vector<TokenId> output_tokens;
  std::vector<float> finetuning_losses;
};

struct RotaryEmbeddingMeta {
  bool apply_rotary_embedding = false;
  float rope_theta = 10000.0f;
  std::string rope_type = "default";
  float factor = 8.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_position_embeddings = 8192;

  RotaryEmbeddingMeta(bool apply_rotary_embedding_ = false,
                      float rope_theta_ = 10000.0f,
                      std::string rope_type_ = "default",
                      float factor_ = 8.0f,
                      float low_freq_factor_ = 1.0f,
                      float high_freq_factor_ = 4.0f,
                      int original_max_position_embeddings_ = 8192)
      : apply_rotary_embedding(apply_rotary_embedding_),
        rope_theta(rope_theta_), rope_type(rope_type_), factor(factor_),
        low_freq_factor(low_freq_factor_), high_freq_factor(high_freq_factor_),
        original_max_position_embeddings(original_max_position_embeddings_) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  RotaryEmbeddingMeta const &meta) {
    os << std::boolalpha // To print bool as true/false instead of 1/0
       << "RotaryEmbeddingMeta {\n"
       << "  apply_rotary_embedding: " << meta.apply_rotary_embedding << ",\n"
       << "  rope_theta: " << meta.rope_theta << ",\n"
       << "  rope_type: \"" << meta.rope_type << "\",\n"
       << "  factor: " << meta.factor << ",\n"
       << "  low_freq_factor: " << meta.low_freq_factor << ",\n"
       << "  high_freq_factor: " << meta.high_freq_factor << ",\n"
       << "  original_max_position_embeddings: "
       << meta.original_max_position_embeddings << "\n"
       << "}";
    return os;
  }
};

std::string join_path(std::vector<std::string> const &paths);

} // namespace FlexFlow
