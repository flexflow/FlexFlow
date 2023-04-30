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

#pragma once

#include <cstdlib>

// #define MAX_SEQ_LEN 1024
// #define BATCH_SIZE 2
#define MAX_SEQ_LEN 20
#define BATCH_SIZE 16
#define MAX_REQUESTS 256

namespace FlexFlow {

struct InferenceResult {
  static int const MAX_NUM_TOKENS = MAX_SEQ_LEN * BATCH_SIZE;
  int results[MAX_NUM_TOKENS];
};

class BatchConfig {
public:
  BatchConfig();
  bool register_new_request(size_t guid,
                            int initial_len,
                            int tokens_to_generate);
  void prepare_next_batch();
  int update_results(InferenceResult const &ir);
  void update_num_active_requests_tokens();
  int num_active_requests() const;
  int num_active_tokens() const;
  void print() const;
  static int const MAX_NUM_REQUESTS = MAX_REQUESTS;
  static int const MAX_NUM_TOKENS = InferenceResult::MAX_NUM_TOKENS;

  //  These are set by update
  int num_tokens, num_requests;
  bool cached_results;

  struct PerRequestInfo {
    size_t token_start_offset;
    size_t num_tokens_in_batch;
    size_t guid;
  };
  struct PerTokenInfo {
    size_t abs_depth_in_request;
    size_t request_index;
    size_t value;
  };
  PerRequestInfo requestsInfo[MAX_NUM_REQUESTS];
  PerTokenInfo tokensInfo[MAX_NUM_TOKENS];

  size_t max_sequence_length[MAX_NUM_REQUESTS];
  bool request_completed[MAX_NUM_REQUESTS];
};

}; // namespace FlexFlow
