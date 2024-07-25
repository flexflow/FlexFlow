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

#include "flexflow/page_manager.h"
#include "flexflow/parallel_ops/parallel_op.h"
#include "flexflow/tokenizers.h"
#include <bitset>
#include <cmath>
#include <filesystem>
#include <future>
#include <iomanip>
#include <new>
#include <random>
#include <stack>
#include <stdexcept>

namespace FlexFlow {
    class LogicalBlock {
public:
    // Constructor
    LogicalBlock(int block_number, int block_size)
        : block_number(block_number), block_size(block_size), num_tokens(0) {
        token_ids.resize(block_size, _BLANK_TOKEN_ID);
    }

    // Method to check if the block is empty
    bool is_empty() const {
        return num_tokens == 0;
    }

    // Method to get the number of empty slots
    int get_num_empty_slots() const {
        return block_size - num_tokens;
    }

    // Method to check if the block is full
    bool is_full() const {
        return num_tokens == block_size;
    }

    // Method to append tokens
    void append_tokens(const std::vector<int>& token_ids_to_append) {
        assert(token_ids_to_append.size() <= get_num_empty_slots());
        std::copy(token_ids_to_append.begin(), token_ids_to_append.end(), token_ids.begin() + num_tokens);
        num_tokens += token_ids_to_append.size();
    }

    // Method to get the list of token ids
    std::vector<int> get_token_ids() const {
        return std::vector<int>(token_ids.begin(), token_ids.begin() + num_tokens);
    }

    // Method to get the last token id
    int get_last_token_id() const {
        assert(num_tokens > 0);
        return token_ids[num_tokens - 1];
    }

private:
    int block_number; 
    int block_size; //how many tokens can be stored in the block
    int num_tokens; //how many tokens are currenty in the block
    std::vector<int> token_ids; //TODO: change to the corresponding token index style
};

class PhysicalTokenBlock {
public:
    // Constructor
    PhysicalTokenBlock(int block_number, int block_size)
        : block_number(block_number), block_size(block_size), ref_count(0) {}

private:
    int block_number;
    int block_size; //how many tokens can be stored in the block
    int ref_count; //how many references to the block， WARNING: but currently we don't need it
};

BlockAllocator::BlockAllocator(Device device, int block_size, int num_blocks)
    : device(device), block_size(block_size), num_blocks(num_blocks) {
    // Initialize the free blocks
    for (int i = 0; i < num_blocks; ++i) {
        PhysicalTokenBlock block(device, i, block_size);
        free_blocks.push_back(block);
    }
}

// Allocate a block
PhysicalTokenBlock BlockAllocator::allocate() {
    if (free_blocks.empty()) {
        throw std::runtime_error("Out of memory! No free blocks are available.");
    }
    PhysicalTokenBlock block = free_blocks.back();
    free_blocks.pop_back();
    block.ref_count = 1;
    return block;
}

// Free a block
void BlockAllocator::free(PhysicalTokenBlock& block) {
    if (block.ref_count == 0) {
        throw std::runtime_error("Double free! Block is already freed.");
    }
    block.ref_count -= 1;
    if (block.ref_count == 0) {
        free_blocks.push_back(block);
    }
}

// Get the number of free blocks
int BlockAllocator::get_num_free_blocks() const {
    return free_blocks.size();
}

PageManager::PageManager(int block_size, int num_blocks)
    : block_size(block_size), num_blocks(num_blocks),
      gpu_allocator(block_size, num_blocks) {}

bool PageManager::can_allocate(const RequestGuid& request_guid) const {
    const Sequence& seq = seq_group.get_seqs().front();
    int num_required_blocks = seq.get_logical_token_blocks().size();
    if (block_sliding_window != -1) {
        num_required_blocks = std::min(num_required_blocks, block_sliding_window);
    }
    int num_free_gpu_blocks = gpu_allocator.get_num_free_blocks();
    return (num_free_gpu_blocks - num_required_blocks >= watermark_blocks);
}

void PageManager::allocate(const RequestGuid& request_guid) {
    const Sequence& seq = seq_group.get_seqs().front();
    BlockTable block_table;

    for (size_t logical_idx = 0; logical_idx < seq.get_logical_token_blocks().size(); ++logical_idx) {
        PhysicalTokenBlock block;
        if (block_sliding_window != -1 && logical_idx >= block_sliding_window) {
            block = block_table[logical_idx % block_sliding_window];
        } else {
            block = gpu_allocator.allocate();
        }
        block.set_ref_count(seq_group.num_seqs());
        block_table.push_back(block);
    }

    for (const Sequence& seq : seq_group.get_seqs()) {
        block_tables[seq.get_seq_id()] = block_table;
    }
}

bool PageManager::can_append_slot(const RequestGuid& request_guid) const {
    int num_free_gpu_blocks = gpu_allocator.get_num_free_blocks();
    int num_seqs = seq_group.num_seqs(SequenceStatus::RUNNING);
    return num_seqs <= num_free_gpu_blocks;
}

std::optional<std::pair<int, int>> PageManager::append_slot(const RequestGuid& request_guid) {
    auto& logical_blocks = seq.get_logical_token_blocks();
    auto& block_table = block_tables[seq.get_seq_id()];

    if (block_table.size() < logical_blocks.size()) {
        if (block_sliding_window != -1 && block_table.size() >= block_sliding_window) {
            block_table.push_back(block_table[block_table.size() % block_sliding_window]);
        } else {
            PhysicalTokenBlock block = gpu_allocator.allocate();
            block_table.push_back(block);
            return std::nullopt;
        }
    }

    PhysicalTokenBlock& last_block = block_table.back();
    if (last_block.get_ref_count() == 1) {
        return std::nullopt;
    } else {
        PhysicalTokenBlock new_block = gpu_allocator.allocate();
        block_table.back() = new_block;
        gpu_allocator.free(last_block);
        return std::make_pair(last_block.get_block_number(), new_block.get_block_number());
    }
}

void PageManager::_free_block_table(const BlockTable& block_table) {
    for (const auto& block : block_table) {
            gpu_allocator.free(block);
        } 
    }
}

void PageManager::free(const RequestGuid& request_guid) {
    if (block_tables.find(seq.get_seq_id()) == block_tables.end()) {
        return;
    }
    auto& block_table = block_tables[seq.get_seq_id()];
    _free_block_table(block_table);
    block_tables.erase(seq.get_seq_id());
}

std::vector<int> PageManager::get_block_table(const RequestGuid& request_guid) const {
    std::vector<int> block_numbers;
    for (const auto& block : block_tables.at(seq.get_seq_id())) {
        block_numbers.push_back(block.get_block_number());
    }
    return block_numbers;
}

int PageManager::get_num_free_blocks() const {
    return gpu_allocator.get_num_free_blocks();
}

BlockTable PageManager::_get_physical_blocks(const RequestGuid& request_guid) const {
    std::unordered_set<PhysicalTokenBlock> blocks;
    for (const Sequence& seq : seq_group.get_seqs()) {
        if (seq.is_finished()) {
            continue;
        }
        auto& block_table = block_tables.at(seq.get_seq_id());
        blocks.insert(block_table.begin(), block_table.end());
    }
    return BlockTable(blocks.begin(), blocks.end());
}

}; //FlexFlow