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

#ifndef _FLEXFLOW_UTILS_PEFT_WEIGHT_ALLOCATOR_H_
#define _FLEXFLOW_UTILS_PEFT_WEIGHT_ALLOCATOR_H_

#include "flexflow/config.h"
#include <mutex>

namespace FlexFlow {

#ifdef DEACODE
class PEFTWeightAllocator {
public:
  PEFTWeightAllocator(void *_base_ptr, size_t _total_size)
      : base_ptr(_base_ptr), total_size(_total_size), sync_offset(0),
        local_offset(_total_size) {}

  inline void *allocate_sync_weights_untyped(PEFTModelID const &peft_model_id,
                                             size_t datalen) {
    const std::lock_guard<std::mutex> lock(peft_weight_allocator_mutex);
    void *ptr = static_cast<char *>(base_ptr) + sync_offset;
    off_t model_sync_weights_offset = sync_offset;
    size_t model_sync_weights_size = datalen;
    if (sync_weights.find(peft_model_id) != sync_weights.end()) {
      // Assert that sync weights for each PEFT model is consecutive
      std::pair<off_t, size_t> offset_and_size = sync_weights[peft_model_id];
      assert(sync_offset == offset_and_size.first + offset_and_size.second);
      model_sync_weights_offset = offset_and_size.first;
      model_sync_weights_size = offset_and_size.second + datalen;
    }
    sync_offset += datalen;
    assert(sync_offset < local_offset);
    sync_weights[peft_model_id] =
        std::make_pair(model_sync_weights_offset, model_sync_weights_size);
    return ptr;
  }

  std::pair<void *, size_t>
      get_sync_weights_ptr_and_size(PEFTModelID const &peft_model_id) {
    const std::lock_guard<std::mutex> lock(peft_weight_allocator_mutex);
    assert(sync_weights.find(peft_model_id) != sync_weights.end());
    std::pair<off_t, size_t> offset_and_size = sync_weights[peft_model_id];
    return std::make_pair(static_cast<char *>(base_ptr) + offset_and_size.first,
                          offset_and_size.second);
  }

  inline void *allocate_local_weights_untyped(PEFTModelID const &peft_model_id,
                                              size_t datalen) {
    const std::lock_guard<std::mutex> lock(peft_weight_allocator_mutex);
    local_offset -= datalen;
    assert(sync_offset < local_offset);
    void *ptr = static_cast<char *>(base_ptr) + local_offset;
    return ptr;
  }

  template <typename DT>
  inline DT *allocate_sync_weights(PEFTModelID const &peft_model_id,
                                   size_t count) {
    return static_cast<DT *>(
        allocate_sync_weights_untyped(peft_model_id, sizeof(DT) * count));
  }

  template <typename DT>
  inline DT *allocate_local_weights(PEFTModelID const &peft_model_id,
                                    size_t count) {
    return static_cast<DT *>(
        allocate_local_weights_untyped(peft_model_id, sizeof(DT) * count));
  }

public:
  void *base_ptr;
  size_t total_size;
  off_t sync_offset, local_offset;
  std::unordered_map<PEFTModelID, std::pair<off_t, size_t>> sync_weights;
  std::mutex peft_weight_allocator_mutex;
};
#endif

class PEFTMemoryManager {
public:
  PEFTMemoryManager(int max_rank_, int max_concurrent_adapters_, int lora_in_dim, int lora_out_dim) : max_rank(max_rank_), max_concurrent_adapters(max_concurrent_adapters_), lora_in_dim(lora_in_dim), lora_out_dim(lora_out_dim) {}

  void allocate_memory();
  void register_peft_model(PEFTModelID const &model_id);
  


  int max_rank, max_concurrent_adapters;
  int lora_in_dim, lora_out_dim;
}

}; // namespace FlexFlow

#endif // _FLEXFLOW_UTILS_PEFT_WEIGHT_ALLOCATOR_H_
