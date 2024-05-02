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

#define DEFAULT_BEAM_WIDTH 1
#define DEFAULT_TARGET_ITERATIONS 3

namespace FlexFlow {

LegionRuntime::Logger::Category log_tree_bc("TreeSearchBatchConfig");

TreeSearchBatchConfig::TreeSearchBatchConfig() : BatchConfig() {}

TreeSearchBatchConfig::TreeSearchBatchConfig(int model_id)
    : BatchConfig(), model_id(model_id) {
  std::cout << "==================\n"
            << "Register Batch Config with Model " << this->model_id
            << std::endl;
}

/* Why do we need this? */
TreeSearchBatchConfig::TreeSearchBatchConfig(TreeSearchBatchConfig const &other,
                                             int model_id)
    : BatchConfig(), model_id(model_id) {}

TreeSearchBatchConfig::~TreeSearchBatchConfig() {}

InferenceMode TreeSearchBatchConfig::get_mode() const {
  return TREE_SEARCH_MODE;
}

std::ostream &
    operator<<(std::ostream &os,
               TreeSearchBatchConfig const &tree_search_batch_config) {
  os << "@@@@@@@@@@@@@@ TreeSearchBatchConfig (mode "
     << tree_search_batch_config.get_mode() << ") @@@@@@@@@@@@@@" << std::endl;
  // Max values
  os << "Max number of requests: "
     << tree_search_batch_config.max_requests_per_batch() << std::endl;
  os << "Max number of tokens: "
     << tree_search_batch_config.max_tokens_per_batch() << std::endl;
  os << "Max sequence length: "
     << tree_search_batch_config.max_sequence_length() << std::endl;
  // Current values
  os << "Number of tokens: " << tree_search_batch_config.num_active_tokens()
     << std::endl;
  os << "Number of requests: " << tree_search_batch_config.num_active_requests()
     << std::endl;
  // Tree Search-specific
  os << "Model ID: " << tree_search_batch_config.model_id << std::endl;
  os << "Max tree depth: " << TreeSearchBatchConfig::MAX_TREE_DEPTH
     << std::endl;
  os << "Max num branch: "
     << TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES << std::endl;

  os << "Per-request info:\n";
  for (int i = 0; i < tree_search_batch_config.max_requests_per_batch(); i++) {
    if (!tree_search_batch_config.request_available[i]) {
      os << "  Request " << i << ":\n";
      os << "    First token depth in request: "
         << tree_search_batch_config.requestsInfo[i]
                .first_token_index_in_request
         << std::endl;
      os << "    First token offset in batch: "
         << tree_search_batch_config.requestsInfo[i].first_token_offset_in_batch
         << std::endl;
      os << "    Number of tokens in batch: "
         << tree_search_batch_config.requestsInfo[i].num_tokens_in_batch
         << std::endl;
      os << "    Request available: "
         << tree_search_batch_config.request_available[i] << std::endl;
    }
  }

  os << "Per-token info:\n";
  for (int i = 0; i < tree_search_batch_config.num_tokens; i++) {
    os << "  Token " << i << ":\n";
    os << "    Absolute depth in request: "
       << tree_search_batch_config.tokensInfo[i].abs_index_in_request
       << std::endl;
    os << "    Request index: "
       << tree_search_batch_config.tokensInfo[i].request_index << std::endl;
    os << "    Token id: " << tree_search_batch_config.tokensInfo[i].token_id
       << std::endl;
  }
  os << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  return os;
}

void TreeSearchBatchConfig::print() const {
  std::cout << *this << std::endl;
}

void TreeSearchBatchConfig::save_to_file(std::string const &filename) const {
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
