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
#include <cassert>
#include "legion.h"

namespace FlexFlow {

LegionRuntime::Logger::Category log_bc("BatchConfig");

BatchConfig::BatchConfig() {
  cached_results = false;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    token_start_idx[i] = 0;
    token_last_available_idx[i] = -1;
    request_completed[i] = true;
    num_processing_tokens[i] = 0;
  }
}

int BatchConfig::update_results(InferenceResult const &ir) {
  cached_results = false;
  int t = 0;
  int completed = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i])
      continue;
    if (num_processing_tokens[i] == 0)
      continue;
    t += num_processing_tokens[i];
    token_start_idx[i] += num_processing_tokens[i];
    if (ir.results[t] == 0) { // TODO: replace this with <EOS>
      log_bc.print("[Done] guid(%zu) final_length(%d)", request_guid[i], token_start_idx[i]);
      request_completed[i] = true;
      token_start_idx[i] = 0;
      token_last_available_idx[i] = -1;
      num_processing_tokens[i] = 0;
      completed ++;
    } else if (token_start_idx[i] >= MAX_SEQUENCE_LENGTH) {
      //Reach maximum request length
      log_bc.print("[Done] guid(%zu) final_length(%d)", request_guid[i], token_start_idx[i]);
      request_completed[i] = true;
      token_start_idx[i] = 0;
      token_last_available_idx[i] = -1;
      num_processing_tokens[i] = 0;
      completed ++;
    } else {
      if (token_start_idx[i] == token_last_available_idx[i] + 1)
        token_last_available_idx[i] ++;
      assert(token_start_idx[i] <= token_last_available_idx[i]);
    }
    num_processing_tokens[i] = 0;
  }
  return completed;
}

bool BatchConfig::register_new_request(size_t guid, int length) {
  cached_results = false;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      log_bc.print("[NewRequest] guid(%zu) length(%d)", guid, length);
      token_start_idx[i] = 0;
      token_last_available_idx[i] = length - 1;
      request_guid[i] = guid;
      num_processing_tokens[i] = 0;
      request_completed[i] = false;
      return true;
    }
  }
  return false;
}

void BatchConfig::prepare_next_batch() {
  cached_results = false;
  int num_tokens = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i])
      continue;
    if (num_tokens + token_last_available_idx[i] - token_start_idx[i] + 1 <= MAX_NUM_TOKENS) {
      num_processing_tokens[i] = token_last_available_idx[i] - token_start_idx[i] + 1;
    } else {
      num_processing_tokens[i] = MAX_NUM_TOKENS - num_tokens;
    }
    num_tokens += num_processing_tokens[i];
  }
  log_bc.print("[NextBatch] num_tokens(%d)", num_tokens);
}

int BatchConfig::num_active_requests() {
  if (cached_results)
    return num_requests;
  num_requests = 0;
  num_tokens = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      num_requests ++;
      num_tokens += num_processing_tokens[i];
    }
  }
  cached_results = true;
  return num_requests;
}

int BatchConfig::num_active_tokens() {
  if (cached_results)
    return num_tokens;
  num_requests = 0;
  num_tokens = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      num_requests ++;
      num_tokens += num_processing_tokens[i];
    }
  }
  cached_results = true;
  return num_tokens;
}

}; // namespace FlexFlow
