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
  PEFTMemoryManager(size_t max_lora_size_, int max_concurrent_adapters_) 
  : max_concurrent_adapters(max_concurrent_adapters_), max_lora_size(max_lora_size_), base_ptr(nullptr) {}

  // allocate memory for all the PEFT adapters for a given layer on a given shard
  void allocate_memory(Memory gpu_mem) {
    // allocate chunk of memory for all the PEFT adapters
    Realm::Rect<1, coord_t> bounds(
        Realm::Point<1, coord_t>(0),
        Realm::Point<1, coord_t>(max_lora_size - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(peftLegionInst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    base_ptr = peftLegionInst.pointer_untyped(0, sizeof(char));
  }
  
  // Returns the slot in memory where the peft model weights are/will be stored. 
  // If the model is not in memory (cache miss), set the cache_miss flag to true.
  void *get_peft_model_handle(PEFTModelID const &model_id, bool *cache_miss) {
    assert(base_ptr != nullptr && "PEFT Memory Manager not initialized");
    assert(lru_hashtable.size() == lru_list.size() &&
           lru_list.size() == peft2mem_slot.size() &&
           "PEFT Memory Manager LRU hashtable/list and/or peft2mem_slot are out of sync");
    // check for cache hit
    if (lru_hashtable.find(model_id) != lru_hashtable.end()) {
      int lru_list_index = lru_hashtable[model_id];
      assert(lru_list[lru_list_index] == model_id &&
             "PEFT Memory Manager LRU hashtable/list are out of sync");
      // move the model to the end of the LRU list
      lru_list.erase(lru_list.begin() + lru_list_index);
      lru_list.push_back(model_id);
      // update the LRU hashtable
      lru_hashtable[model_id] = lru_list.size() - 1;
      // get memory slot
      assert(peft2mem_slot.find(model_id) != peft2mem_slot.end() && "PEFT Memory Manager peft2mem_slot is out of sync");
      *cache_miss = false;
    } else {
      // cache miss
      // check if you need to evict
      bool need_to_evict = lru_list.size() == max_concurrent_adapters;
      int mem_slot = -1;
      if (need_to_evict) {
        // evict the least recently used model
        PEFTModelID lru_model_id = lru_list[0];
        lru_list.erase(lru_list.begin());
        lru_hashtable.erase(lru_model_id);
        mem_slot = peft2mem_slot[lru_model_id];
        peft2mem_slot.erase(lru_model_id);
      } else {
        mem_slot = lru_list.size();
      }
      // update the LRU list and hashtable
      lru_list.push_back(model_id);
      lru_hashtable[model_id] = lru_list.size() - 1;
      // update the memory slot
      peft2mem_slot[model_id] = mem_slot;
      *cache_miss = true;
    }
    return static_cast<char *>(base_ptr) + peft2mem_slot[model_id]*max_lora_size;
  }

  int max_concurrent_adapters;
  size_t max_lora_size;
  Realm::RegionInstance peftLegionInst;
  void *base_ptr;
  std::unordered_map<PEFTModelID, int> lru_hashtable;
  std::vector<PEFTModelID> lru_list; // head = least recently used, tail=most recently used
  std::unordered_map<PEFTModelID, int> peft2mem_slot;
}

}; // namespace FlexFlow

#endif // _FLEXFLOW_UTILS_PEFT_WEIGHT_ALLOCATOR_H_
