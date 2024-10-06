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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/lora_linear_params.h"
// #include <mutex>

namespace FlexFlow {

#ifdef DEADCODE
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

struct LoraLinearWeight {
  // weights
  void *w0_ptr, *w1_ptr;
  // gradients
  void *w0_grad_ptr, *w1_grad_ptr;
  // activations
  void *input_activation;
  void *low_rank_activation;
  // v values for SGD optimizer (when using momentum)
  void *w0_v_values_ptr, *w1_v_values_ptr;
  LoraLinearWeight(void *w0 = nullptr,
                   void *w1 = nullptr,
                   void *w0_grad = nullptr,
                   void *w1_grad = nullptr,
                   void *w0_v_values = nullptr,
                   void *w1_v_values = nullptr,
                   void *low_rank_activation_ = nullptr,
                   void *input_activation_ = nullptr)
      : w0_ptr(w0), w1_ptr(w1), w0_grad_ptr(w0_grad), w1_grad_ptr(w1_grad),
        w0_v_values_ptr(w0_v_values), w1_v_values_ptr(w1_v_values),
        low_rank_activation(low_rank_activation_),
        input_activation(input_activation_) {}
};

void init_peft_weight_wrapper(LoraLinearWeight const &weight,
                              int in_dim,
                              int out_dim,
                              int rank,
                              DataType dt,
                              int seed);

class PEFTMemoryManager {
public:
  PEFTMemoryManager(Legion::Memory gpu_mem_,
                    int max_rank_,
                    int max_concurrent_adapters_,
                    int max_peft_tokens_,
                    int in_dim_,
                    int out_dim_,
                    int num_shards_,
                    int shard_id_,
                    std::string const &lora_layername_substr_,
                    DataType dt_)
      : gpu_mem(gpu_mem_), max_concurrent_adapters(max_concurrent_adapters_),
        max_rank(max_rank_), in_dim(in_dim_), out_dim(out_dim_),
        num_shards(num_shards_), shard_id(shard_id_),
        max_peft_tokens(max_peft_tokens_),
        lora_layername_substr(lora_layername_substr_), dt(dt_),
        base_ptr(nullptr), finetuning_ptr(nullptr),
        finetuning_model_id(PEFTModelID::NO_ID) {
    max_lora_size =
        data_type_size(dt) * (max_rank * in_dim + max_rank * out_dim);
    assert(max_concurrent_adapters > 0 &&
           "PEFT Memory Manager max_concurrent_adapters must be > 0");
    assert(max_lora_size > 0 &&
           "PEFT Memory Manager max_lora_size must be > 0");
    allocate_inference_memory();
    // finetuning memory is allocated upon the first finetuning request, so we
    // can skip for inference-only workloads
  }

  // allocate memory for all the PEFT adapters for a given layer on a given
  // shard
  void allocate_inference_memory();
  // allocate memory for the PEFT adapter for a finetuning request for a given
  // layer and shard
  void allocate_finetuning_memory();

  LoraLinearWeight get_peft(PEFTModelID const &model_id,
                            LoraLinearConfig const &lora_config);
  void check_ft_model_id(PEFTModelID const &model_id);

private:
  // Check if the PEFT adapter for the given model is in memory. If not, sets
  // the cache_miss flag to true. If this is the first finetuning request,
  // allocate memory for the finetuning adapter.
  void get_finetuning_slot(PEFTModelID const &model_id, bool *cache_miss);
  // Returns the slot in memory where the peft model weights are/will be stored.
  // If the model is not in memory (cache miss), set the cache_miss flag to
  // true.
  int get_inference_peft_slot(PEFTModelID const &model_id, bool *cache_miss);
  void load_peft_model(LoraLinearWeight &weight,
                       LoraLinearConfig const &lora_config);
  LoraLinearWeight get_inference_peft(PEFTModelID const &model_id,
                                      LoraLinearConfig const &lora_config);
  LoraLinearWeight get_finetuning_peft(PEFTModelID const &model_id,
                                       LoraLinearConfig const &lora_config);

  // Legion memory management apparatus
  Legion::Memory gpu_mem;
  Realm::RegionInstance peftLegionInst;
  void *base_ptr, *finetuning_ptr;
  // Size and shapes
  int max_concurrent_adapters;
  int max_rank;
  int max_lora_size;
  int in_dim, out_dim, num_shards, shard_id;
  int max_peft_tokens;
  // LRU cache apparatus
  std::unordered_map<PEFTModelID, int> lru_hashtable;
  std::vector<PEFTModelID>
      lru_list; // head = least recently used, tail=most recently used
  std::unordered_map<PEFTModelID, int> peft2mem_slot;
  // Miscellanea
  std::string lora_layername_substr;
  DataType dt;
  PEFTModelID finetuning_model_id;
};

} // namespace FlexFlow

#endif // _FLEXFLOW_UTILS_PEFT_WEIGHT_ALLOCATOR_H_
