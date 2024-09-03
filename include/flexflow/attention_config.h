/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#ifndef _FLEXFLOW_ATTENTION_CONFIG_H_
#define _FLEXFLOW_ATTENTION_CONFIG_H_
#include "flexflow/batch_config.h"

namespace FlexFlow {

constexpr uint32_t kPagesize = 64;

inline int round_up_pages(int const num_elements) {
  return (num_elements + kPagesize - 1) / kPagesize;
}

#define DISPATCH_HEADDIM(head_dim, HEAD_DIM, ...)                              \
  switch (head_dim) {                                                          \
    case 64: {                                                                 \
      constexpr size_t HEAD_DIM = 64;                                          \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 128: {                                                                \
      constexpr size_t HEAD_DIM = 128;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 256: {                                                                \
      constexpr size_t HEAD_DIM = 256;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "Unsupported head_dim: " << head_dim;                         \
      throw std::invalid_argument(err_msg.str());                              \
    }                                                                          \
  }

class AttentionMetaData {
public:
  AttentionMetaData() {
    num_q_heads_ = 0;
    num_kv_heads_ = 0;
    head_dim_ = 0;
    q_indptr = nullptr;
    kv_indptr = nullptr;
    kv_indices = nullptr;
    kv_last_page_len = nullptr;
    qk_indptr = nullptr;
    custom_mask = nullptr;
    workspace = nullptr;
    workspace_size = 0;
    float_workspace = nullptr;
    float_workspace_size = 0;
    int_workspace = nullptr;
    int_workspace_size = 0;
    mem_size_ = 0;
    enabled_ = false;
  }
  AttentionMetaData(AttentionMetaData const &rhs) {
    num_q_heads_ = rhs.num_q_heads_;
    num_kv_heads_ = rhs.num_kv_heads_;
    head_dim_ = rhs.head_dim_;
    q_indptr = rhs.q_indptr;
    kv_indptr = rhs.kv_indptr;
    kv_indices = rhs.kv_indices;
    kv_last_page_len = rhs.kv_last_page_len;
    qk_indptr = rhs.qk_indptr;
    custom_mask = rhs.custom_mask;
    workspace = rhs.workspace;
    workspace_size = rhs.workspace_size;
    float_workspace = rhs.float_workspace;
    float_workspace_size = rhs.float_workspace_size;
    int_workspace = rhs.int_workspace;
    int_workspace_size = rhs.int_workspace_size;
    mem_size_ = rhs.mem_size_;
    enabled_ = rhs.enabled_;
    decode_handler_collections = rhs.decode_handler_collections;
    prompt_handler_collections = rhs.prompt_handler_collections;
  }

  size_t mem_size() {
    if (mem_size_ > 0) {
      return mem_size_;
    }
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages =
        round_up_pages(BatchConfig::max_spec_tree_token_num() +
                       BatchConfig::max_sequence_length());
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);
    size_t custom_mask_size = BatchConfig::max_requests_per_batch() *
                              ((BatchConfig::max_spec_tree_token_num() *
                                    (BatchConfig::max_spec_tree_token_num() +
                                     BatchConfig::max_sequence_length()) +
                                7) /
                               8);

    float_workspace_size = 128 * 1024 * 1024; // 128 MB
    int_workspace_size = 8 * 1024 * 1024;     // 8 MB
    workspace_size =
        float_workspace_size + int_workspace_size; // float + int workspace

    mem_size_ = sizeof(int32_t) * indices_size +
                sizeof(uint8_t) * custom_mask_size + workspace_size;
    return mem_size_;
  }

  void assign_address(void *ptr, int size) {
    if (ptr == nullptr) {
      q_indptr = nullptr;
      kv_indptr = nullptr;
      kv_indices = nullptr;
      kv_last_page_len = nullptr;
      qk_indptr = nullptr;
      custom_mask = nullptr;
      workspace = nullptr;
      float_workspace = nullptr;
      int_workspace = nullptr;
      return;
    }
    assert(size >= mem_size() &&
           "Insufficient memory size for attention metadata");
    size_t batch_size = BatchConfig::max_requests_per_batch();
    size_t max_num_pages =
        round_up_pages(BatchConfig::max_spec_tree_token_num() +
                       BatchConfig::max_sequence_length());
    size_t indices_size = std::max(
        (batch_size + 1) * 4 + max_num_pages * batch_size, 1ul * 1024 * 1024);
    size_t custom_mask_size = BatchConfig::max_requests_per_batch() *
                              ((BatchConfig::max_spec_tree_token_num() *
                                    (BatchConfig::max_spec_tree_token_num() +
                                     BatchConfig::max_sequence_length()) +
                                7) /
                               8);

    q_indptr = static_cast<int32_t *>(ptr);
    kv_indptr = q_indptr + batch_size + 1;
    kv_indices = kv_indptr + batch_size + 1;
    kv_last_page_len = kv_indices + max_num_pages * batch_size;
    qk_indptr = kv_last_page_len + batch_size + 1;
    custom_mask = static_cast<uint8_t *>(ptr) + sizeof(int32_t) * indices_size;
    workspace = static_cast<void *>(static_cast<uint8_t *>(ptr) +
                                    sizeof(int32_t) * indices_size +
                                    sizeof(uint8_t) * custom_mask_size);
    float_workspace = workspace;
    int_workspace = static_cast<void *>(static_cast<uint8_t *>(workspace) +
                                        float_workspace_size);
  }

  void set_num_q_heads(uint32_t const num_q_heads) {
    num_q_heads_ = num_q_heads;
  }
  void set_num_kv_heads(uint32_t const num_kv_heads) {
    num_kv_heads_ = num_kv_heads;
  }
  void set_head_dim(uint32_t const head_dim) {
    head_dim_ = head_dim;
  }
  uint32_t num_q_heads() const {
    return num_q_heads_;
  }
  uint32_t num_kv_heads() const {
    return num_kv_heads_;
  }
  uint32_t head_dim() const {
    return head_dim_;
  }

  void set_enabled(bool const enabled) {
    enabled_ = enabled;
  }
  bool enabled() const {
    return enabled_;
  }

  uint32_t num_q_heads_;
  uint32_t num_kv_heads_;
  uint32_t head_dim_;

  int32_t *q_indptr;
  int32_t *kv_indptr;
  int32_t *kv_indices;
  int32_t *kv_last_page_len;
  int32_t *qk_indptr;
  uint8_t *custom_mask;
  void *workspace;
  size_t workspace_size;
  void *float_workspace;
  size_t float_workspace_size;
  void *int_workspace;
  size_t int_workspace_size;

  size_t mem_size_;

  // batchsize -> handler
  bool enabled_;
  std::unordered_map<int, void *> decode_handler_collections;
  std::unordered_map<int, void *> prompt_handler_collections;
};
} // namespace FlexFlow

#endif // _FLEXFLOW_ATTENTION_CONFIG_H_
