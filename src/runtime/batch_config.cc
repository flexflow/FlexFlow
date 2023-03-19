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
    token_start_idx[i] = 0;
    token_last_available_idx[i] = -1;
    request_completed[i] = true;
    num_processing_tokens[i] = 0;
    max_sequence_length[i] = 0;
  }
  token2ids.num_samples = 0;
  for (int i = 0; i < MAX_NUM_TOKENS; i++) {
    token2ids.guids[i] = SIZE_MAX;
    token2ids.token_indexes[i].request_index = SIZE_MAX;
    token2ids.token_indexes[i].token_position = SIZE_MAX;
  }
  update_num_active_requests_tokens();
}

int BatchConfig::update_results(InferenceResult const &ir) {
  cached_results = false;
  int t = 0;
  int completed = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      continue;
    }
    if (num_processing_tokens[i] == 0) {
      continue;
    }
    t += num_processing_tokens[i];
    token_start_idx[i] += num_processing_tokens[i];
    if (token_start_idx[i] >= max_sequence_length[i]
        // || ir.results[t] == 0 TODO: replace this with <EOS>
    ) {
      log_bc.print("[Done] guid(%zu) final_length(%d)",
                   request_guid[i],
                   token_start_idx[i]);
      request_completed[i] = true;
      token_start_idx[i] = 0;
      token_last_available_idx[i] = -1;
      num_processing_tokens[i] = 0;
      completed++;
    } else {
      if (token_start_idx[i] == token_last_available_idx[i] + 1) {
        token_last_available_idx[i]++;
      }
      assert(token_start_idx[i] <= token_last_available_idx[i]);
    }
    num_processing_tokens[i] = 0;
  }
  update_num_active_requests_tokens();
  return completed;
}

bool BatchConfig::register_new_request(size_t guid,
                                       int initial_length,
                                       int tokens_to_generate) {
  cached_results = false;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      log_bc.print("[NewRequest] guid(%zu) length(%d)", guid, initial_length);
      token_start_idx[i] = 0;
      token_last_available_idx[i] = initial_length - 1;
      max_sequence_length[i] = initial_length + tokens_to_generate;
      request_guid[i] = guid;
      num_processing_tokens[i] = 0;
      request_completed[i] = false;
      update_num_active_requests_tokens();
      return true;
    }
  }
  update_num_active_requests_tokens();
  return false;
}

void BatchConfig::prepare_next_batch() {
  cached_results = false;
  int count = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      continue;
    }
    if (num_tokens + token_last_available_idx[i] - token_start_idx[i] + 1 <=
        MAX_NUM_TOKENS) {
      num_processing_tokens[i] =
          token_last_available_idx[i] - token_start_idx[i] + 1;
    } else {
      num_processing_tokens[i] = MAX_NUM_TOKENS - num_tokens;
    }
    count += num_processing_tokens[i];
  }
  update_num_active_requests_tokens();
  log_bc.print("[NextBatch] num_tokens(%d)", count);
}

void BatchConfig::update_num_active_requests_tokens() {
  num_requests = 0;
  num_tokens = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      num_requests++;
      for (int j = 0; j < num_processing_tokens[i]; j++) {
        token2ids.guids[num_tokens] = request_guid[i];
        token2ids.token_indexes[num_tokens].token_position =
            token_start_idx[i] + j;
        token2ids.token_indexes[num_tokens].request_index = i;
        num_tokens++;
      }
    }
  }
  token2ids.num_samples = num_tokens;
  cached_results = true;
}

int BatchConfig::num_active_requests() const {
  if (cached_results) {
    return num_requests;
  } else {
    assert(false &&
           "some BatchConfig functions updated requests but didn't call "
           "update_num_active_requests_tokens() before exit");
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

}; // namespace FlexFlow
