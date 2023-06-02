/* Copyright 2023 CMU, Stanford, Facebook, LANL
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

#include "flexflow/batch_config.h"
#include "legion.h"
#include <cassert>
#include <climits>

namespace FlexFlow {

LegionRuntime::Logger::Category log_tree_bc("TreeVerifyBatchConfig");

TreeVerifyBatchConfig::TreeVerifyBatchConfig() : BatchConfig() {}

TreeVerifyBatchConfig::~TreeVerifyBatchConfig() {}

InferenceMode TreeVerifyBatchConfig::get_mode() const {
  return TREE_VERIFY_MODE;
}

void TreeVerifyBatchConfig::print() const {
  std::cout << "@@@@@@@@@@@@@@ TreeVerifyBatchConfig (mode " << get_mode()
            << ") @@@@@@@@@@@@@@" << std::endl;
  std::cout << "Max number of requests: " << MAX_NUM_REQUESTS << std::endl;
  std::cout << "Max number of tokens: " << MAX_NUM_TOKENS << std::endl;
  std::cout << "Number of tokens: " << num_tokens << std::endl;
  std::cout << "Number of requests: " << num_active_requests() << std::endl;
  // std::cout << "Cached results: " << cached_results << std::endl;

  std::cout << "Per-request info:\n";
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      std::cout << "  Request " << i << ":\n";
      std::cout << "    Token start offset: "
                << requestsInfo[i].token_start_offset << std::endl;
      std::cout << "    Number of tokens in batch: "
                << requestsInfo[i].num_tokens_in_batch << std::endl;
      std::cout << "    GUID: " << requestsInfo[i].request_guid << std::endl;
      std::cout << "    Max sequence length: "
                << requestsInfo[i].max_sequence_length << std::endl;
      std::cout << "    Request completed: " << request_completed[i]
                << std::endl;
    }
  }

  std::cout << "Per-token info:\n";
  for (int i = 0; i < num_tokens; i++) {
    std::cout << "  Token " << i << ":\n";
    std::cout << "    Absolute depth in request: "
              << tokensInfo[i].abs_depth_in_request << std::endl;
    std::cout << "    Request index: " << tokensInfo[i].request_index
              << std::endl;
    std::cout << "    Token id: " << tokensInfo[i].token_id << std::endl;
  }

  std::cout << "Tokens to commit info:\n";
  for (int i = 0; i < num_tokens_to_commit; i++) {
    std::cout << "  Token " << i << ":\n";
    std::cout << "    token_index: " << committed_tokens[i].token_index
              << std::endl;
    std::cout << "    request_index: " << committed_tokens[i].request_index
              << std::endl;
    std::cout << "    token_depth: " << committed_tokens[i].token_depth
              << std::endl;
  }

  std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            << std::endl;
}

}; // namespace FlexFlow
