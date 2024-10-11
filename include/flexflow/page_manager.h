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

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/config.h"
#include "flexflow/utils/file_loader.h"
#include <future>
#include <mutex>
#include <tokenizers_cpp.h>
#include <deque>

namespace FlexFlow {

using TokenId = BatchConfig::TokenId;

/**
 * @class LogicalTokenBlock
 * @brief A class to represent a logical block of tokens similar to virtual memory address
 */
class LogicalTokenBlock {
public:
    using TokenId = BatchConfig::TokenId;

    // Constructor
    LogicalTokenBlock(int block_number, uint32_t block_size);

    // Method to check if the block is empty
    bool is_empty() const;

    // Method to check if the block is full
    bool is_full() const;

    // Method to get the number of empty slots
    int get_num_empty_slots() const;

    // Method to get the number of allocated slots
    int get_num_alloc_slots() const;

    // Used to clean up the spec tokens in a block since these spec tokens may not be committed after use
    void reset_num_spec_tokens();

    // Method to append tokens
    void append_tokens(const std::vector<TokenId>& token_ids_to_append, bool committed);

    int get_num_tokens() const { return num_tokens; }
    int get_num_commit_tokens() const { return num_commit_tokens; }
    int get_num_spec_tokens() const { return num_spec_tokens; }

    std::vector<TokenId> get_token_ids() const;

private:
    int block_number; // the index of the logical token block
    int block_size; // the size of the block
    int num_tokens; // the number of tokens currently stored in the block
    int num_commit_tokens; // the number of tokens inside this block that are already committed
    int num_spec_tokens; // the number of tokens inside this block that are speculative tokens, which is stored temporarily

    std::vector<TokenId> token_ids; //store the token ids in a order that corresponds to the inference sequence
};

/**
 * @class PhysicalTokenBlock
 * @brief A class to represent a physical block of tokens similar to physical memory address
 * It keeps track of the location of the tokens stored on GPU memory
 */
class PhysicalTokenBlock {
public:
    // Constructor
    PhysicalTokenBlock(int block_number, int block_size);

    // Method to get the block number
    int get_block_number() const { return block_number; }
    void incr_ref_count() { ref_count++; }
    void decr_ref_count() { ref_count--; }
    int ref_count; // reference count, TODO: move to private

private:
    int block_number; // the index of the physical token block
    int block_size; // the size of the block
};

/**
 * @class BlockAllocator
 * @brief A Block Manager that is reponsible for maintaining a pool of free blocks
 */
class BlockAllocator {
public:
    // Constructor
    BlockAllocator(int block_size, int num_total_blocks);

    // Allocate a block
    PhysicalTokenBlock allocate();

    // Free a block
    void free(PhysicalTokenBlock& block);

    // Get the number of free blocks
    int get_num_free_blocks() const;

private:
    int block_size;
    int num_total_blocks;
    std::deque<PhysicalTokenBlock> free_blocks;
};

/*
* @class PageManager
* @brief A wrapper class that manages the kv cache allocation status
* notice that all the layers of model will share the same page manager because the position of kv cache will be the same
*/
class PageManager {
public:
    // Get the singleton instance of the PageManager as it will be shared in multiple places
    static PageManager *get_page_manager();
    using BlockTable = std::vector<PhysicalTokenBlock>;
    using RequestGuid = BatchConfig::RequestGuid;
    PageManager(int block_size, int num_total_blocks);


    int allocate_one_block(const RequestGuid& request_guid);
    void free_request(const RequestGuid& request_guid);
    void free_multiple_blocks(const RequestGuid& request_guid, int num_blocks);
    std::vector<int> get_block_table_indices(const RequestGuid& request_guid) const;

    
    void free_block_table(BlockTable& block_table);
private:
    int block_size; // the size of the block
    int num_total_blocks; // the total number of blocks
    BlockAllocator block_allocator;
    std::unordered_map<RequestGuid, BlockTable> block_tables;

    int get_num_total_free_blocks() const;
    int get_num_allocated_blocks(const RequestGuid& request_guid) const;
};

}; // namespace FlexFlow