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
  bool spec_sample = false;
  float temperature = 0.8;
  // top-p renormalization
  float topp = 0.6;
  // top-k renormalization
  int topk = 16;
  GenerationConfig(bool _do_sample = false,
                   float _temperature = 0.8,
                   float _topp = 0.6,
                   bool _spec_sample = false,
                   int _topk = 16)
      : do_sample(_do_sample), temperature(_temperature), topp(_topp),
        spec_sample(_spec_sample), topk(_topk) {
    assert(temperature > 0.0);
    assert(topk <= BatchConfig::MAX_K_LOGITS);
  }
};

struct GenerationRequest {
  std::string prompt;
  double slo_ratio;

  GenerationRequest(std::string const &prompt_, double slo_ratio_)
      : prompt(prompt_), slo_ratio(slo_ratio_) {}

  GenerationRequest(std::string const &prompt_)
      : prompt(prompt_), slo_ratio(1.0) {}
};

struct GenerationResult {
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;
  RequestGuid guid;
  std::string input_text;
  std::string output_text;
  std::vector<TokenId> input_tokens;
  std::vector<TokenId> output_tokens;
};

#include <string>
#include <vector>

std::string join_path(std::vector<std::string> const &paths);

} // namespace FlexFlow
