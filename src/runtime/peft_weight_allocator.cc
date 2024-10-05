#include "peft_weight_allocator.h"

namespace FlexFlow {

void PEFTMemoryManager::allocate_inference_memory() {
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

void PEFTMemoryManager::allocate_finetuning_memory() {
    size_t ft_size = max_lora_size*3; // weights, gradients, momentum values
    ft_size += max_peft_tokens*(in_dim+rank); // input, low-rank activations
    // allocate chunk of memory for PEFT adapter
    Realm::Rect<1, coord_t> bounds(
        Realm::Point<1, coord_t>(0),
        Realm::Point<1, coord_t>(ft_size - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(peftLegionInst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    finetuning_ptr = peftLegionInst.pointer_untyped(0, sizeof(char));
}

void PEFTMemoryManager::get_finetuning_slot(PEFTModelID const &model_id, bool *cache_miss) {
    if (finetuning_ptr == nullptr) {
      allocate_finetuning_memory();
    }
    assert(finetuning_ptr != nullptr && "PEFT Memory Manager finetuning_ptr is null");
    *cache_miss = (model_id.id != finetuning_model_id.id);
}

int PEFTMemoryManager::get_inference_peft_slot(PEFTModelID const &model_id, bool *cache_miss) {
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
    assert(peft2mem_slot.find(model_id) != peft2mem_slot.end() && "PEFT Memory Manager peft2mem_slot is out of sync");
    int slot = peft2mem_slot[model_id];
    assert(slot >= 0 && slot < max_concurrent_adapters && "PEFT Memory Manager peft2mem_slot is out of bounds");
    return slot;
}

template <typename DT>
void load_peft_from_file(DT *ptr,
                         size_t num_rows,
                         size_t num_columns,
                         int num_shards,
                         int shard_id,
                         std::string filepath) {
  std::ifstream in(filepath, std::ios::in | std::ios::binary);
  if (!in.good()) {
    printf("Could not open file: %s\n", filepath.c_str());
  }
  assert(in.good() && "incorrect weight file path");

  // HuggingFace dims (serialized in row-major order)
  //    lora_A: [rank, intermediate_dim]
  //    lora_B: [hidden_dim, rank]
  // FlexFlow dims (serialized in column-major order)
  //    lora_A: [intermediate_dim, rank]
  //    lora_B: [rank, out_dim]
  // Tensor parallelism: shard lora_A along intermediate_dim, replicate lora_B
  assert(num_rows % num_shards == 0);
  size_t chunk_size = num_rows / num_shards;
  size_t offset = (num_shards > 1) ? shard_id * chunk_size : 0;

  // Allocate memory for the weight shard
  std::vector<DT> host_array(chunk_size * num_columns);
  // Read the chunk
  size_t total_size_read = 0;
  for (int i = 0; i < num_columns; ++i) {
    in.seekg((i * num_rows + offset) * sizeof(DT));
    in.read(reinterpret_cast<char *>(host_array.data() + i * chunk_size),
            chunk_size * sizeof(DT));
    total_size_read += in.gcount();
  }
  // Check weight shard size
  size_t expected_data_size = chunk_size * num_columns * sizeof(DT);
  if (total_size_read != expected_data_size) {
    printf("load weight data error: expected %lu bytes, got: %lu bytes, data "
           "size: %lu\n",
           expected_data_size,
           total_size_read,
           sizeof(DT));
    assert(false);
  }
  assert(host_array.size() == chunk_size * num_columns);
  // Copy weight to device memory
  copy_tensor_host_to_dev(ptr, host_array.data(), chunk_size * num_columns);
  in.close();
}

void PEFTMemoryManager::load_peft_model(LoraLinearWeight &weight, LoraLinearConfig const &lora_config) {
    // Load weights
    assert(weight.w0_ptr != nullptr && weight.w1_ptr != nullptr "PEFT Memory Manager weight ptr null");
    int w0_num_elements = lora_config.rank * in_dim;
    int w1_num_elements = lora_config.rank * out_dim;
    // values below represent total weight sizes before sharding. Lora B is not
    // sharded.
    int lora_A_num_rows = in_dim * num_shards;
    int lora_A_num_cols = lora_config.rank;
    int lora_B_num_rows = lora_config.rank;
    int lora_B_num_cols = out_dim;
    int lora_A_num_shards = num_shards;
    int lora_B_num_shards = 1;
    if (lora_config.init_lora_weights) {
        // initialize weights randomly
        int seed = 0;
        init_peft_weight_wrapper(weight, in_dim, out_dim, lora_config.rank, dt, seed);
    } else {
        // load weights from file
        std::string weights_folder_filepath = join_path({
            lora_config.cache_folder,
            "weights",
            lora_config.peft_model_id,
            dt == DT_FLOAT ? "full-precision" : "half-precision",
        });
        std::string w0_filepath = join_path(
            {weights_folder_filepath, lora_layername_substr + "_A.weight"});
        std::string w1_filepath = join_path(
            {weights_folder_filepath, lora_layername_substr + "_B.weight"});
        if (dt == DT_FLOAT) {
            std::cout << "Loading LORA weight "
                        << lora_layername_substr + "_A.weight"
                        << ", num_rows: " << lora_A_num_rows
                        << ", num_cols: " << lora_A_num_cols
                        << ", num_shards: " << lora_A_num_shards
                        << ", shard_id: " << shard_id << std::endl;
            load_peft_from_file((float *)weight.w0_ptr,
                                lora_A_num_rows,
                                lora_A_num_cols,
                                lora_A_num_shards,
                                shard_id,
                                w0_filepath);
            std::cout << "Loading LORA weight "
                        << lora_layername_substr + "_B.weight"
                        << ", num_rows: " << lora_B_num_rows
                        << ", num_cols: " << lora_B_num_cols
                        << ", num_shards: " << lora_B_num_shards
                        << ", shard_id: " << shard_id << std::endl;
            load_peft_from_file((float *)weight.w1_ptr,
                                lora_B_num_rows,
                                lora_B_num_cols,
                                lora_B_num_shards,
                                shard_id,
                                w1_filepath);
        } else if (dt == DT_HALF) {
            std::cout << "Loading LORA weight "
                        << lora_layername_substr + "_A.weight"
                        << ", num_rows: " << lora_A_num_rows
                        << ", num_cols: " << lora_A_num_cols
                        << ", num_shards: " << lora_A_num_shards
                        << ", shard_id: " << shard_id << std::endl;
            load_peft_from_file((half *)weight.w0_ptr,
                                lora_A_num_rows,
                                lora_A_num_cols,
                                lora_A_num_shards,
                                shard_id,
                                w0_filepath);
            std::cout << "Loading LORA weight "
                        << lora_layername_substr + "_B.weight"
                        << ", num_rows: " << lora_B_num_rows
                        << ", num_cols: " << lora_B_num_cols
                        << ", num_shards: " << lora_B_num_shards
                        << ", shard_id: " << shard_id << std::endl;
            load_peft_from_file((half *)weight.w1_ptr,
                                lora_B_num_rows,
                                lora_B_num_cols,
                                lora_B_num_shards,
                                shard_id,
                                w1_filepath);
        } else {
            assert(false && "Data type not supported");
        }
    }
}

LoraLinearWeight PEFTMemoryManager::get_inference_peft(PEFTModelID const &model_id, LoraLinearConfig const &lora_config) {
    assert(model_id != PEFTModelID::NO_ID && "PEFT Model ID is not set");
    bool cache_miss;
    int mem_slot = get_inference_peft_slot(model_id, &cache_miss);
    int w0_num_elements = lora_config.rank * in_dim;
    int data_size = data_type_size(dt);
    LoraLinearWeight result;
    result.w0_ptr = static_cast<char *>(base_ptr) + mem_slot * max_lora_size;
    result.w1_ptr = result.w0_ptr + w0_num_elements * data_size;
    if (cache_miss) {
      load_peft_model(result, lora_config);
    }
    return result;
}

LoraLinearWeight PEFTMemoryManager::get_finetuning_peft(PEFTModelID const &model_id, LoraLinearConfig const &lora_config) {
    assert(model_id != PEFTModelID::NO_ID && "PEFT Model ID is not set");
    bool cache_miss = get_finetuning_slot(model_id);
    int w0_num_elements = lora_config.rank * in_dim;
    int w1_num_elements = lora_config.rank * out_dim;
    int data_size = data_type_size(dt);
    LoraLinearWeight result;
    result.w0_ptr = finetuning_ptr;
    result.w1_ptr = result.w0_ptr + w0_num_elements*data_size;
    result.w0_grad_ptr = result.w1_ptr + w1_num_elements*data_size;
    result.w1_grad_ptr = result.w0_grad_ptr + w0_num_elements*data_size;
    result.w0_v_values_ptr = result.w1_grad_ptr + w1_num_elements*data_size;
    result.w1_v_values_ptr = result.w0_v_values_ptr + w0_num_elements*data_size;
    result.input_activation = result.w1_v_values_ptr + w1_num_elements*data_size; // max_peft_tokens*in_dim
    result.low_rank_activation = result.input_activation + max_peft_tokens*in_dim*data_size; // max_peft_tokens*rank
    if (cache_miss) {
      load_peft_model(result, lora_config);
    }
    return result;
}

LoraLinearWeight PEFTMemoryManager::get_peft(PEFTModelID const &model_id, LoraLinearConfig const &lora_config) {
    if (lora_config.trainable) {
      return get_finetuning_peft(model_id, lora_config);
    } else {
      return get_inference_peft(model_id, lora_config);
    }
}

}; // namespace FlexFlow