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

LegionRuntime::Logger::Category log_bc("BatchConfig");
using Legion::Future;
using Legion::Memory;

BatchConfig::BatchConfig() : num_tokens(0) {
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    requestsInfo[i].token_start_offset = 0;
    requestsInfo[i].num_tokens_in_batch = 0;
    request_completed[i] = true;
  }
  for (int i = 0; i < MAX_NUM_TOKENS; i++) {
    tokensInfo[i].abs_depth_in_request = 0;
    tokensInfo[i].request_index = 0;
    tokensInfo[i].token_id = 0;
  }
}

/*static*/
BatchConfig const *BatchConfig::from_future(BatchConfigFuture const &future) {
  BatchConfig const *bc = static_cast<BatchConfig const *>(
      Future(future).get_buffer(Memory::SYSTEM_MEM));
  // Check future size
  if (bc->get_mode() == INC_DECODING_MODE) {
    assert(Future(future).get_untyped_size() == sizeof(BatchConfig));
  } else if (bc->get_mode() == BEAM_SEARCH_MODE) {
    assert(Future(future).get_untyped_size() == sizeof(BeamSearchBatchConfig));
  } else if (bc->get_mode() == TREE_VERIFY_MODE) {
    assert(Future(future).get_untyped_size() == sizeof(TreeVerifyBatchConfig));
  } else {
    assert(false && "Unsupported inference mode");
  }
  return bc;
}

InferenceMode BatchConfig::get_mode() const {
  return INC_DECODING_MODE;
}

int BatchConfig::num_active_requests() const {
  int num_requests = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      num_requests++;
    }
  }
  return num_requests;
}

int BatchConfig::num_active_tokens() const {
  return num_tokens;
}

void BatchConfig::print() const {
  std::cout << "@@@@@@@@@@@@@@ Batch Config (mode " << get_mode()
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
  std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            << std::endl;
}

}; // namespace FlexFlow
