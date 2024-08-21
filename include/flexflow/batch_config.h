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

namespace FlexFlow {

class InferenceResult;

using BatchConfigFuture = Legion::Future;
using InferenceResultFuture = Legion::Future;

class BatchConfig {
public:
  using RequestGuid = size_t;
  using TokenId = int;
  BatchConfig(InferenceMode inference_mode = INC_DECODING_MODE,
              int model_id = 0);
  BatchConfig(BatchConfig const &other);
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
  inline static int const MAX_NUM_REQUESTS = 64;
  inline static int const MAX_NUM_TOKENS = 1024;
  inline static int const MAX_SPEC_TREE_TOKEN_NUM = 128;
  inline static int const MAX_SPECULATIVE_TREE_BRANCHES = 4;
  inline static int const MAX_TREE_DEPTH = 16;
  inline static int const MAX_TREE_WIDTH = 64;
  inline static int const MAX_K_LOGITS = 16;

  int num_tokens = 0;
  int num_available_requests = 0;
  bool prompt_phase = false;
  int num_tokens_to_commit = 0;
  int model_id;
  InferenceMode inference_mode;

  struct PerRequestInfo {
    int first_token_index_in_request = -1;
    int first_token_offset_in_batch = -1;
    int num_tokens_in_batch = 0;

    // page attention: we need some additional attention information here to allocate physical blocks in load_batch_config
    // TODO: might need to add more fields here
    int32_t num_kv_pages; //number of kv pages used
    int32_t kv_last_page_len;
  };

  struct PerTokenInfo {
    TokenId token_id = -1;
    // Difference between the two:
    // abs_index_in_request: non-tree cache size + index in the flattened
    // speculative tree
    // abs_depth_in_request: non_tree cache size + depth in the speculative tree
    int abs_index_in_request = -1;
    int abs_depth_in_request = -1;
    int request_index = -1;
  };

  std::vector<int32_t> page_indices; //the indices for each page

  struct CommittedTokensInfo {
    int index_in_kv_cache = -1; // the index in the temporary key-value cache
    int request_index = -1;     // request index in the batch
    int token_depth = -1; // position of the token in the request's sequence
  };

  class BitMask {
  public:
    class Bitset {
    public:
      Bitset() : bits{0} {}

      Bitset(Bitset const &other) {
        // Copy the entire array of bits from 'other' to this object
        std::copy(
            std::begin(other.bits), std::end(other.bits), std::begin(bits));
      }

      void set_bit(size_t pos) {
        size_t idx = pos / 64; // Find the index in the array
        size_t bit = pos % 64; // Find the bit position within the uint64_t
        bits[idx] |= (1ULL << bit);
      }

      bool test_bit(size_t pos) const {
        size_t idx = pos / 64;
        size_t bit = pos % 64;
        return (bits[idx] & (1ULL << bit)) != 0;
      }

      void clear() {
        std::fill(std::begin(bits), std::end(bits), 0);
      }

      uint64_t bits[MAX_SPEC_TREE_TOKEN_NUM / 64];
    };

    Bitset bit_mask[MAX_SPEC_TREE_TOKEN_NUM];
    // the number of generated tokens before the speculation tree (excluding the
    // prompt tokens)
    int non_tree_cache_size = 0;
    // Tree size or prompt size. Because the prefilling phase and the decoding
    // phase are separated, we only need one field to store the size of the tree
    // or the prompt.
    int tree_or_prompt_size = 0;
    int current_layer_size = 0;

    BitMask() = default;

    BitMask(BitMask const &other) {
      non_tree_cache_size = other.non_tree_cache_size;
      tree_or_prompt_size = other.tree_or_prompt_size;
      current_layer_size = other.current_layer_size;
      for (int i = 0; i < MAX_SPEC_TREE_TOKEN_NUM; i++) {
        bit_mask[i] = other.bit_mask[i];
      }
    }

    void clear_bitmask() {
      // Clear bit_mask but keep the other fields
      for (int i = 0; i < MAX_SPEC_TREE_TOKEN_NUM; i++) {
        bit_mask[i].clear();
      }
    }
  };

  BitMask causalMask[MAX_NUM_REQUESTS];
  PerRequestInfo requestsInfo[MAX_NUM_REQUESTS];
  std::vector<int32_t> requestsIndices[MAX_NUM_REQUESTS]; //for kv cache
  PerTokenInfo tokensInfo[MAX_NUM_TOKENS];
  CommittedTokensInfo committed_tokens[MAX_NUM_TOKENS];
  bool request_available[MAX_NUM_REQUESTS];
};

struct InferenceResult {
  int num_token_ids;
  int num_gumbel_logits;
  BatchConfig::TokenId
      token_ids[BatchConfig::MAX_NUM_TOKENS * BatchConfig::MAX_K_LOGITS];
  float probs[BatchConfig::MAX_NUM_TOKENS * BatchConfig::MAX_K_LOGITS];
  float gumbel_logits[BatchConfig::MAX_NUM_TOKENS *
                      BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
  InferenceResult() : num_token_ids(0), num_gumbel_logits(0) {}
  InferenceResult(InferenceResult const &other);
};

}; // namespace FlexFlow
