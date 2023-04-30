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

BatchConfig::BatchConfig() {
  cached_results = false;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    requestsInfo[i].token_start_offset = 0;
    requestsInfo[i].num_tokens_in_batch = 0;
    request_completed[i] = true;
  }
  for (int i = 0; i < MAX_NUM_TOKENS; i++) {
    tokensInfo[i].abs_depth_in_request = SIZE_MAX;
    tokensInfo[i].request_index = SIZE_MAX;
    tokensInfo[i].value = SIZE_MAX;
  }
  update_num_active_requests_tokens();
}

int BatchConfig::update_results(InferenceResult const &ir) {
  cached_results = false;
  int completed = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      continue;
    }
    assert(requestsInfo[i].num_tokens_in_batch > 0);
    int processed_tokens = requestsInfo[i].token_start_offset +
                           requestsInfo[i].num_tokens_in_batch;
    if (processed_tokens >= max_sequence_length[i]
        // || ir.results[t] == 0 TODO: replace this with <EOS>
    ) {
      log_bc.print("[Done] guid(%zu) final_length(%d)",
                   requestsInfo[i].guid,
                   processed_tokens);
      request_completed[i] = true;
      requestsInfo[i].num_tokens_in_batch = 0;
      requestsInfo[i].token_start_offset = 0;
      completed++;
    } else {
      requestsInfo[i].token_start_offset += requestsInfo[i].num_tokens_in_batch;
      requestsInfo[i].num_tokens_in_batch = 1;
    }
  }
  update_num_active_requests_tokens();
  return completed;
}

bool BatchConfig::register_new_request(size_t guid,
                                       int initial_len,
                                       int tokens_to_generate) {
  cached_results = false;
  assert(initial_len > 0 && tokens_to_generate > 0);
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      log_bc.print("[NewRequest] guid(%zu) length(%d)", guid, initial_len);
      requestsInfo[i].token_start_offset = 0;
      requestsInfo[i].num_tokens_in_batch = initial_len;
      requestsInfo[i].guid = guid;
      max_sequence_length[i] = initial_len + tokens_to_generate;
      request_completed[i] = false;
      update_num_active_requests_tokens();
      return true;
    }
  }
  update_num_active_requests_tokens();
  return false;
}

void BatchConfig::prepare_next_batch() {
  assert(cached_results);
  assert(num_requests > 0 && num_tokens > 0);
  log_bc.print("[NextBatch] num_tokens(%d)", num_tokens);
}

void BatchConfig::update_num_active_requests_tokens() {
  num_requests = 0;
  num_tokens = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      num_requests++;
      for (int j = 0; j < requestsInfo[i].num_tokens_in_batch; j++) {
        int start_idx = requestsInfo[i].token_start_offset;
        tokensInfo[num_tokens].abs_depth_in_request = start_idx + j;
        tokensInfo[num_tokens].request_index = i;
        num_tokens++;
      }
    }
  }
  cached_results = true;
}

int BatchConfig::num_active_requests() const {
  if (cached_results) {
    return num_requests;
  } else {
    assert(false &&
           "some BatchConfig functions updated requests but didn't call "
           "() before exit");
  }
}

int BatchConfig::num_active_tokens() const {
  if (cached_results) {
    return num_tokens;
  } else {
    assert(false &&
           "some BatchConfig functions updated requests but didn't call "
           "update_num_active_requests_tokens() before exit");
  }
}

void BatchConfig::print() const {
  std::cout << "Max number of requests: " << MAX_NUM_REQUESTS << std::endl;
  std::cout << "Max number of tokens: " << MAX_NUM_TOKENS << std::endl;
  std::cout << "Number of tokens: " << num_tokens << std::endl;
  std::cout << "Number of requests: " << num_requests << std::endl;
  std::cout << "Cached results: " << cached_results << std::endl;

  std::cout << "Per-request info:\n";
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      std::cout << "  Request " << i << ":\n";
      std::cout << "    Token start offset: "
                << requestsInfo[i].token_start_offset << std::endl;
      std::cout << "    Number of tokens in batch: "
                << requestsInfo[i].num_tokens_in_batch << std::endl;
      std::cout << "    GUID: " << requestsInfo[i].guid << std::endl;
      std::cout << "    Max sequence length: " << max_sequence_length[i]
                << std::endl;
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
  }
}

}; // namespace FlexFlow
