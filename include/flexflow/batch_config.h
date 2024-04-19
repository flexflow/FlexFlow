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
class SsmInferenceResult;

using BatchConfigFuture = Legion::Future;
using InferenceResultFuture = Legion::Future;
using TreeSearchBatchConfigFuture = Legion::Future;
using TreeVerifyBatchConfigFuture = Legion::Future;
using SsmInferenceResultFuture = Legion::Future;

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

  int num_tokens;
  int num_available_requests;

  struct PerRequestInfo {
    int first_token_index_in_request;
    int first_token_offset_in_batch;
    int num_tokens_in_batch;
  };
  struct PerTokenInfo {
    TokenId token_id;
    int abs_index_in_request;
    int request_index;
    // For SSM KV cache commitment
    int kv_cache_dest_index = -1;
  };

  class BitMask {
    class Bitset {
    public:
      Bitset() : bits{0} {}

      Bitset(Bitset const &other) {
        // Copy the entire array of bits from 'other' to this object
        std::memcpy(bits, other.bits, sizeof(bits));
      }

      void set_bit(size_t pos) {
        size_t idx = pos / 64; // Find the index in the array
        size_t bit = pos % 64; // Find the bit position within the uint64_t
        bits[idx] |= (1ULL << bit);
      }

      void reset_bit(size_t pos) {
        size_t idx = pos / 64;
        size_t bit = pos % 64;
        bits[idx] &= ~(1ULL << bit);
      }

      bool test_bit(size_t pos) const {
        size_t idx = pos / 64;
        size_t bit = pos % 64;
        return (bits[idx] & (1ULL << bit)) != 0;
      }

    private:
      uint64_t bits[MAX_SPEC_TREE_TOKEN_NUM / 8]; // Array to hold 256 bits
    };

  public:
    Bitset bit_mask[MAX_SPEC_TREE_TOKEN_NUM];
    // the number of generated tokens before the speculation tree (excluding the
    // prompt tokens)
    int non_tree_cache_size = 0;
    // current tree size
    int tree_size = 0;
    int current_layer_size = 0;
    // input length-> prompt/root
    int prompt_size = 0;
    BitMask() = default;
    BitMask(BitMask const &other) {
      non_tree_cache_size = other.non_tree_cache_size;
      tree_size = other.tree_size;
      current_layer_size = other.current_layer_size;
      prompt_size = other.prompt_size;
      for (int i = 0; i < MAX_SPEC_TREE_TOKEN_NUM; i++) {
        bit_mask[i] = other.bit_mask[i];
      }
    }
  };

  BitMask causalMask[MAX_NUM_REQUESTS];
  PerRequestInfo requestsInfo[MAX_NUM_REQUESTS];
  PerTokenInfo tokensInfo[MAX_NUM_TOKENS];

  bool request_available[MAX_NUM_REQUESTS];
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

class TreeSearchBatchConfig : public BatchConfig {
public:
  TreeSearchBatchConfig();
  TreeSearchBatchConfig(int model_id);
  TreeSearchBatchConfig(TreeSearchBatchConfig const &other, int model_id);
  InferenceMode get_mode() const;

  ~TreeSearchBatchConfig();

  friend std::ostream &operator<<(std::ostream &os,
                                  TreeSearchBatchConfig const &bc);
  void print() const;
  void save_to_file(std::string const &filename) const;
  int current_depth() const;
  int get_speculative_request_num() const;

  inline static int const MAX_SPECULATIVE_TREE_BRANCHES = 3;
  inline static int const MAX_TREE_DEPTH = 16;

  // how many requests is in speculative phase
  int current_depth = 0;
  int model_id;
};

class SsmInferenceResult : public InferenceResult {
public:
  BatchConfig::TokenId
      token_ids[MAX_NUM_TOKENS *
                TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
  float probs[MAX_NUM_TOKENS *
              TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
  int parent_id[MAX_NUM_TOKENS *
                TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
};

}; // namespace FlexFlow
