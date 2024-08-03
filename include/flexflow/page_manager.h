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
#include "flexflow/utils/file_loader.h"
#include <future>
#include <mutex>
#include <tokenizers_cpp.h>
#include <deque>

namespace FlexFlow {

class LogicalTokenBlock {
public:
    // Constructor
    LogicalTokenBlock(int block_number, int block_size);

    // Method to check if the block is empty
    bool is_empty() const;

    // Method to get the number of empty slots
    int get_num_empty_slots() const;

    // Method to check if the block is full
    bool is_full() const;

    // Method to append tokens
    void append_tokens(const std::vector<int>& token_ids_to_append);

    // Method to get the list of token ids
    std::vector<int> get_token_ids() const;

    // Method to get the last token id
    int get_last_token_id() const;

    int block_number;
    int block_size;
    int num_tokens;
    std::vector<int> token_ids;
};

class PhysicalTokenBlock {
public:
    // Constructor
    PhysicalTokenBlock(int block_number, int block_size);

    int ref_count;
    int block_number;
    int block_size;
};

class BlockAllocator {
public:
    // Constructor
    BlockAllocator(int block_size, int num_blocks);

    // Allocate a block
    PhysicalTokenBlock allocate();

    // Free a block
    void free(PhysicalTokenBlock& block);

    // Get the number of free blocks
    int get_num_free_blocks() const;

private:
    int block_size;
    int num_blocks;
    std::deque<PhysicalTokenBlock> free_blocks;
};


class PageManager {
public:
    static PageManager *get_page_manager();
    using BlockTable = std::vector<PhysicalTokenBlock>;
    using RequestGuid = BatchConfig::RequestGuid;
    PageManager(int block_size, int num_total_blocks);

    bool prefill(const RequestGuid& request_guid, const std::vector<int>& token_ids);
    bool allocate(const RequestGuid& request_guid);
    void free(const RequestGuid& request_guid);

    int get_num_free_blocks() const;

    std::vector<int> get_block_table_indices(const RequestGuid& request_guid) const;

private:
    int block_size;
    int num_total_blocks;

    BlockAllocator gpu_allocator;
    bool can_prefill(const RequestGuid& request_guid, const std::vector<int>& token_ids);
    bool can_allocate(const RequestGuid& request_guid) const;
    std::unordered_map<int, BlockTable> block_tables;
    void _free_block_table(BlockTable& block_table);
};

}; // namespace FlexFlow