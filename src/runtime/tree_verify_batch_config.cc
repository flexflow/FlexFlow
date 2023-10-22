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
#include "flexflow/request_manager.h"
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

std::ostream &operator<<(std::ostream &os, TreeVerifyBatchConfig const &bc) {
  os << "@@@@@@@@@@@@@@ TreeVerifyBatchConfig (mode " << bc.get_mode()
     << ") @@@@@@@@@@@@@@" << std::endl;
  // Max values
  os << "Max number of requests: " << bc.max_requests_per_batch() << std::endl;
  os << "Max number of tokens: " << bc.max_tokens_per_batch() << std::endl;
  os << "Max sequence length: " << bc.max_sequence_length() << std::endl;
  // Current values
  os << "Number of tokens: " << bc.num_active_tokens() << std::endl;
  os << "Number of requests: " << bc.num_active_requests() << std::endl;
  os << "Number of tokens to commit: " << bc.num_tokens_to_commit << std::endl;

  os << "Per-request info:\n";
  for (int i = 0; i < bc.max_requests_per_batch(); i++) {
    if (!bc.request_completed[i]) {
      os << "  Request " << i << ":\n";
      os << "    First token depth in request: "
         << bc.requestsInfo[i].first_token_depth_in_request << std::endl;
      os << "    First token offset in batch: "
         << bc.requestsInfo[i].first_token_offset_in_batch << std::endl;
      os << "    Number of tokens in batch: "
         << bc.requestsInfo[i].num_tokens_in_batch << std::endl;
      os << "    GUID: " << bc.requestsInfo[i].request_guid << std::endl;
      os << "    Max sequence length: "
         << bc.requestsInfo[i].max_sequence_length << std::endl;
      os << "    Request completed: " << bc.request_completed[i] << std::endl;
      os << "    Request running: " << bc.request_running[i] << std::endl;
    }
  }

  os << "Per-token info:\n";
  for (int i = 0; i < bc.num_tokens; i++) {
    os << "  Token " << i << ":\n";
    os << "    Absolute depth in request: "
       << bc.tokensInfo[i].abs_depth_in_request << std::endl;
    os << "    Request index: " << bc.tokensInfo[i].request_index << std::endl;
    os << "    Token id: " << bc.tokensInfo[i].token_id << std::endl;
  }

  os << "Tokens to commit info:\n";
  for (int i = 0; i < bc.num_tokens_to_commit; i++) {
    os << "  Token " << i << ":\n";
    os << "    token_index: " << bc.committed_tokens[i].token_index
       << std::endl;
    os << "    request_index: " << bc.committed_tokens[i].request_index
       << std::endl;
    os << "    token_depth: " << bc.committed_tokens[i].token_depth
       << std::endl;
  }

  os << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  return os;
}

void TreeVerifyBatchConfig::print() const {
  std::cout << *this << std::endl;
}

void TreeVerifyBatchConfig::save_to_file(std::string const &filename) const {
  std::ofstream outputFile(filename);
  if (outputFile.is_open()) {
    outputFile << *this << std::endl;
    outputFile.close();
  } else {
    std::cerr << "Error: Unable to open the batch config output file: "
              << filename << std::endl;
    assert(false);
  }
}

}; // namespace FlexFlow
