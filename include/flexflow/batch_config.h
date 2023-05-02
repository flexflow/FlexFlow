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

#include <cstddef>
#include <cstdlib>

// #define MAX_SEQ_LEN 1024
// #define BATCH_SIZE 2
#define MAX_SEQ_LEN 20
// #define BATCH_SIZE 16
// #define MAX_REQUESTS 256

namespace FlexFlow {

class InferenceResult;

class BatchConfig {
public:
  using RequestGuid = size_t;
  using TokenId = int;
  BatchConfig();
  bool register_new_request(size_t guid,
                            int initial_len,
                            int tokens_to_generate);
  void prepare_next_batch();
  int update_results(InferenceResult const *ir);
  void update_num_active_requests_tokens();
  int num_active_requests() const;
  int num_active_tokens() const;
  void print() const;
  static int const MAX_NUM_REQUESTS = 8;
  static int const MAX_NUM_TOKENS = 64;

  //  These are set by update
  int num_tokens;

  struct PerRequestInfo {
    int token_start_offset; // input[token_start_offset * data_dim] is the first token
    int num_tokens_in_batch; // tokens from input[token_start_offset * data_dim : (token_start_offset + num_token_in_batch) * data_dim]
    int max_sequence_length;
    RequestGuid request_guid;
  };
  struct PerTokenInfo {
    int abs_depth_in_request;
    int request_index;
    TokenId token_id;
  };
  PerRequestInfo requestsInfo[MAX_NUM_REQUESTS];
  PerTokenInfo tokensInfo[MAX_NUM_TOKENS];

  // size_t max_sequence_length[MAX_NUM_REQUESTS];
  bool request_completed[MAX_NUM_REQUESTS];
};

struct InferenceResult {
  static int const MAX_NUM_TOKENS = BatchConfig::MAX_NUM_TOKENS;
  BatchConfig::TokenId token_ids[MAX_NUM_TOKENS];
};

class BeamSearchBatchConfig : public BatchConfig {
public:  
  BeamSearchBatchConfig();
  BeamSearchBatchConfig(size_t beam_width, size_t target_iterations);

  ~BeamSearchBatchConfig();

  void print() const;
  bool done() const;
  
  size_t beam_width;
  size_t target_iterations;
  static int const MAX_BEAM_WIDTH = 8;
  static int const MAX_BEAM_DEPTH = 8;

  struct BeamSearchPerRequestInfo {
    int token_start_offset; // input[token_start_offset * data_dim] is the first token
    int num_tokens_in_batch; // tokens from input[token_start_offset * data_dim : (token_start_offset + num_token_in_batch) * data_dim]
    int max_sequence_length;
    RequestGuid request_guid;
    bool request_completed;
  };

  struct BeamSearchPerTokenInfo : public PerTokenInfo {
    int token_id;
    int request_index;
    int abs_depth_in_request; 
    int parent_token_id; // parent token id in the previous iteration
    float cum_log_prob;  // cumulative log probability so far
    size_t num_out_beam; // number of beams take this token as output
  };

  BeamSearchPerRequestInfo beamRequestsInfo[MAX_NUM_REQUESTS];
  BeamSearchPerTokenInfo beamTokenInfo[MAX_NUM_TOKENS];

private:
  size_t current_iteration;
};

}; // namespace FlexFlow
