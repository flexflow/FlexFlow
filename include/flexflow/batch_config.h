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
#define BATCH_SIZE 32
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
  // static int const MAX_SEQUENCE_LENGTH = MAX_SEQ_LEN;
  //  These are set by update
  int num_tokens, num_requests;
  int num_beam_tokens, num_beam_requests;
  bool cached_results;
  int token_start_idx[MAX_NUM_REQUESTS]; // index of first token in a request
                                         // that should be processed in the
                                         // current batch/iteration
  int token_last_available_idx
      [MAX_NUM_REQUESTS]; // last valid token index in a request. This includes
                          // both the prompt and generated tokens
  int num_processing_tokens[MAX_NUM_REQUESTS]; // a request's number of tokens
                                               // being processed in the current
                                               // batch/iteration
  size_t initial_length[MAX_NUM_REQUESTS];
  size_t max_sequence_length[MAX_NUM_REQUESTS];

  struct token_idxs {
    size_t request_index;  // the index within the BatchConfig of the request
                           // that the token belongs to

    size_t sub_request_index;  // the index within the BatchConfig of the sub request                       
    size_t token_position; // the index indicating the position of each token
                           // within its request
    size_t initial_length;
  };

  struct SampleIdxs {
    size_t num_samples;
    size_t guids[InferenceResult::MAX_NUM_TOKENS]; // the guid of the request
                                                   // each token belongs to
    token_idxs token_indexes[InferenceResult::MAX_NUM_TOKENS];
  };

  //<request_id, sub_req_num = corresponding beam width>
  std::unordered_map<size_t, int> sub_requests;

  SampleIdxs token2ids;
  size_t request_guid[MAX_NUM_REQUESTS];
  bool request_completed[MAX_NUM_REQUESTS];
};

}; // namespace FlexFlow
