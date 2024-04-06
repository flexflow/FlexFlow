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

#include "flexflow/ffconst.h"
#include "legion.h"
#include <cstddef>
#include <cstdlib>

// #define MAX_SEQ_LEN 1024
// #define BATCH_SIZE 2
// #define BATCH_SIZE 16
// #define MAX_REQUESTS 256

namespace FlexFlow {

class InferenceResult;
class BeamInferenceResult;

using BatchConfigFuture = Legion::Future;
using InferenceResultFuture = Legion::Future;
using BeamSearchBatchConfigFuture = Legion::Future;
using TreeVerifyBatchConfigFuture = Legion::Future;
using BeamInferenceResultFuture = Legion::Future;

class BatchConfig {
public:
  using RequestGuid = size_t;
  using TokenId = int;
  BatchConfig();
  int num_active_requests() const;
  int num_active_tokens() const;
  static int max_requests_per_batch();
  static int max_tokens_per_batch();
  static int max_verify_tokens_per_batch();
  static int max_spec_tree_token_num();
  static int max_sequence_length();
  friend std::ostream &operator<<(std::ostream &os, BatchConfig const &bc);
  void print() const;
  void save_to_file(std::string const &filename) const;
  virtual InferenceMode get_mode() const;
  static BatchConfig const *from_future(BatchConfigFuture const &future);
  // Maximum possible values for different parameters
  // These maximum values are used for copying BatchConfig
  // across workers
  static int const MAX_NUM_REQUESTS = 64;
  static int const MAX_NUM_TOKENS = 1024;
  static int const MAX_SPEC_TREE_TOKEN_NUM = 64;

  //  Set by update
  int num_tokens;
  // number of tokens in prompt phase, start offset of tokens in inc_decoding
  // phase. num_tokens - num_prompt_tokens = num_generation_tokens;
  int num_generation_tokens;

  struct PerRequestInfo {
    int first_token_depth_in_request;
    int first_token_offset_in_batch;
    int num_tokens_in_batch;
    int max_sequence_length;

    // request id in batch config:
    int batch_config_request_id;
    bool prompt_phase = false;
    RequestGuid request_guid;
  };
  struct PerTokenInfo {
    int abs_depth_in_request;
    int request_index;
    TokenId token_id;
  };

  struct BitMask {
    unsigned long long mask[MAX_SPEC_TREE_TOKEN_NUM] = {0};

    // how many tokens before the tree, every sub requests need this part of
    // cache
    int non_tree_cache_size = 0;

    // current tree size
    int tree_size = 0;

    int this_layer_size = 0;

    // input length-> prompt/root
    int prompt_size = 0;
  };

  BitMask causalMask[MAX_NUM_REQUESTS];
  PerRequestInfo requestsInfo[MAX_NUM_REQUESTS];
  PerTokenInfo tokensInfo[MAX_NUM_TOKENS];

  bool request_completed[MAX_NUM_REQUESTS];
  bool request_running[MAX_NUM_REQUESTS];
};

class TreeVerifyBatchConfig : public BatchConfig {
public:
  TreeVerifyBatchConfig();
  ~TreeVerifyBatchConfig();
  InferenceMode get_mode() const;
  friend std::ostream &operator<<(std::ostream &os,
                                  TreeVerifyBatchConfig const &bc);
  void print() const;
  void save_to_file(std::string const &filename) const;
  struct CommittedTokensInfo {
    int token_index;   // the index of the token in the previous batch
    int request_index; // request index in the batch
    int token_depth;   // position of the token in the request's sequence
  };

  int num_tokens_to_commit;
  CommittedTokensInfo committed_tokens[MAX_NUM_TOKENS];
};

struct InferenceResult {
  static int const MAX_NUM_TOKENS = BatchConfig::MAX_NUM_TOKENS;
  BatchConfig::TokenId token_ids[MAX_NUM_TOKENS];
};

class BeamSearchBatchConfig : public BatchConfig {
public:
  BeamSearchBatchConfig();
  BeamSearchBatchConfig(int model_id);
  BeamSearchBatchConfig(size_t beam_width, size_t target_iterations);
  BeamSearchBatchConfig(BeamSearchBatchConfig const &other, int model_id);
  InferenceMode get_mode() const;

  ~BeamSearchBatchConfig();

  friend std::ostream &operator<<(std::ostream &os,
                                  BeamSearchBatchConfig const &bc);
  void print() const;
  void save_to_file(std::string const &filename) const;
  bool done() const;
  int max_beam_depth_all_requests() const;
  int current_depth_all_requests() const;
  int get_speculative_request_num() const;

  size_t beam_width;
  size_t target_iterations;

  // how many requests is in speculative phase
  int speculative_request_num = 0;
  inline static int const MAX_BEAM_WIDTH = 3;
  inline static int const MAX_BEAM_DEPTH = 8;

  // maximum tree branches for a request
  inline static int const MAX_SPECULATIVE_TREE_BRANCHES = 3;

  int model_id;

  struct BeamSearchPerRequestInfo {
    int beam_size;
    int current_depth = -1;
    int max_depth = MAX_BEAM_DEPTH;

    BatchConfig::TokenId
        tokens[BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    float probs[BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    int parent_id[BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    int sub_request_num;
  };

  struct BeamSearchPerTokenInfo {
    int sub_request_index;
  };

  BeamSearchPerRequestInfo beamRequestsInfo[MAX_NUM_REQUESTS];
  BeamSearchPerTokenInfo
      beamTokenInfo[MAX_NUM_TOKENS +
                    MAX_SPEC_TREE_TOKEN_NUM * MAX_NUM_REQUESTS];

  int sub_requests[MAX_NUM_REQUESTS];

private:
  size_t current_iteration;
};

struct BeamInferenceResult {
  static int const MAX_NUM_TOKENS = BatchConfig::MAX_NUM_TOKENS;
  BatchConfig::TokenId
      token_ids[MAX_NUM_TOKENS *
                BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
  float probs[MAX_NUM_TOKENS *
              BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
  int parent_id[MAX_NUM_TOKENS *
                BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
};

}; // namespace FlexFlow
