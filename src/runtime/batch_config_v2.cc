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

#include "flexflow/batch_config_v2.h"
#include "legion.h"
#include <cassert>
#include <climits>

namespace FlexFlow {

LegionRuntime::Logger::Category log_bcv2("BatchConfigV2");

BatchConfigV2::BatchConfigV2() {
  cached_results = false;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    requestsInfo[i].token_start_offset = 0;
    requestsInfo[i].num_tokens_in_batch = 0;
    request_completed[i] = true;
    // token_start_idx[i] = 0;
    // token_last_available_idx[i] = -1;

    // num_processing_tokens[i] = 0;
    // max_sequence_length[i] = 0;
    // initial_length[i] = 0;
  }
  //   token2ids.num_samples = 0;
  for (int i = 0; i < MAX_NUM_TOKENS; i++) {
    tokensInfo[i].abs_depth_in_request = SIZE_MAX;
    tokensInfo[i].request_index = SIZE_MAX;
    // token2ids.guids[i] = SIZE_MAX;
    // token2ids.token_indexes[i].request_index = SIZE_MAX;
    // token2ids.token_indexes[i].token_position = SIZE_MAX;
    // token2ids.token_indexes[i].initial_length = SIZE_MAX;
  }
  update_num_active_requests_tokens();
}

int BatchConfigV2::update_results(InferenceResultV2 const &ir) {
  cached_results = false;
  // int tokens_processed = 0;
  int completed = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      continue;
    }
    assert(requestsInfo[i].num_tokens_in_batch > 0);
    int processed_tokens = requestsInfo[i].token_start_offset +=
        requestsInfo[i].num_tokens_in_batch;
    if (processed_tokens >= max_sequence_length[i]
        // || ir.results[t] == 0 TODO: replace this with <EOS>
    ) {
      log_bcv2.print("[Done] guid(%zu) final_length(%d)",
                     request_guid[i],
                     processed_tokens);
      request_completed[i] = true;
      requestsInfo[i].num_tokens_in_batch = 0;
      requestsInfo[i].token_start_offset = 0;
      //   token_last_available_idx[i] = -1;
      //   num_processing_tokens[i] = 0;
      completed++;
    } else {
      //   if (token_start_idx[i] == token_last_available_idx[i] + 1) {
      //     token_last_available_idx[i]++;
      //     num_processing_tokens[i] = 1; // incremental phase
      //   } else {
      //     assert(false);
      //   }
      requestsInfo[i].token_start_offset += requestsInfo[i].num_tokens_in_batch;
      requestsInfo[i].num_tokens_in_batch = 1;
    }
  }
  update_num_active_requests_tokens();
  return completed;
}

bool BatchConfigV2::register_new_request(size_t guid,
                                         int initial_len,
                                         int tokens_to_generate) {
  cached_results = false;
  assert(initial_len > 0 && tokens_to_generate > 0);
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      log_bcv2.print("[NewRequest] guid(%zu) length(%d)", guid, initial_len);
      requestsInfo[i].token_start_offset = 0;
      requestsInfo[i].num_tokens_in_batch = initial_len;
      //   token_start_idx[i] = 0;
      //   token_last_available_idx[i] = initial_len - 1;
      max_sequence_length[i] = initial_len + tokens_to_generate;
      //   initial_length[i] = initial_len;
      request_guid[i] = guid;
      //   num_processing_tokens[i] = 0;
      request_completed[i] = false;
      update_num_active_requests_tokens();
      return true;
    }
  }
  update_num_active_requests_tokens();
  return false;
}

void BatchConfigV2::prepare_next_batch() {
  cached_results = false;
  int count = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (request_completed[i]) {
      continue;
    }

    if (num_tokens + requestsInfo[i].num_tokens_in_batch <= MAX_NUM_TOKENS) {
      // do nothing, delete it later
    } else {
      requestsInfo[i].num_tokens_in_batch = MAX_NUM_TOKENS - num_tokens;
    }
    // if (num_tokens + token_last_available_idx[i] - token_start_idx[i] + 1 <=
    //     MAX_NUM_TOKENS) {
    //   num_processing_tokens[i] =
    //       token_last_available_idx[i] - token_start_idx[i] + 1;
    // } else {
    //   num_processing_tokens[i] = MAX_NUM_TOKENS - num_tokens;
    // }
    count += requestsInfo[i].num_tokens_in_batch;
  }
  update_num_active_requests_tokens();
  log_bcv2.print("[NextBatch] num_tokens(%d)", count);
}

void BatchConfigV2::update_num_active_requests_tokens() {
  num_requests = 0;
  num_tokens = 0;
  for (int i = 0; i < MAX_NUM_REQUESTS; i++) {
    if (!request_completed[i]) {
      num_requests++;
      for (int j = 0; j < requestsInfo[i].num_tokens_in_batch; j++) {
        int start_idx = requestsInfo[i].token_start_offset;
        tokensInfo[num_tokens].abs_depth_in_request = start_idx + j;
        tokensInfo[num_tokens].request_index = i;
        tokensInfo[num_tokens].guid = request_guid[i];
        // token2ids.guids[num_tokens] = request_guid[i];
        // token2ids.token_indexes[num_tokens].token_position =
        //     token_start_idx[i] + j;
        // token2ids.token_indexes[num_tokens].request_index = i;
        // token2ids.token_indexes[num_tokens].initial_length =
        // initial_length[i];
        num_tokens++;
      }
    }
  }
  //   token2ids.num_samples = num_tokens;
  cached_results = true;
}

int BatchConfigV2::num_active_requests() const {
  if (cached_results) {
    return num_requests;
  } else {
    assert(false &&
           "some BatchConfig functions updated requests but didn't call "
           "() before exit");
  }
}

}; // namespace FlexFlow
