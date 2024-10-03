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

namespace FlexFlow {

// For all runtime functions, they share a single page manager for pages information
PageManager *page_manager_singleton = nullptr;

LogicalTokenBlock::LogicalTokenBlock(int block_number, uint32_t block_size)
    : block_number(block_number), block_size(block_size), num_tokens(0), num_commit_tokens(0), num_spec_tokens(0) {
    }

bool LogicalTokenBlock::is_empty() const {
    assert(num_spec_tokens == 0 && num_commit_tokens == 0);
    assert(num_tokens <= block_size);
    return num_tokens == 0;
}

int LogicalTokenBlock::get_num_empty_slots() const {
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);
    return block_size - num_tokens;
}

int LogicalTokenBlock::get_num_alloc_slots() {
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);
    return num_tokens;
}

bool LogicalTokenBlock::is_full() const {
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);
    return num_tokens == block_size;
}

void LogicalTokenBlock::reset_num_spec_tokens(){
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);

    num_tokens -= num_spec_tokens;
    num_spec_tokens = 0;

    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);
}

void LogicalTokenBlock::append_tokens(const std::vector<TokenId>& token_ids_to_append, bool committed) {
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);
    if (num_tokens + token_ids_to_append.size() > block_size) {
        throw std::runtime_error("Block is full! Cannot append more tokens.");
    }
    token_ids.insert(token_ids.end(), token_ids_to_append.begin(), token_ids_to_append.end());
    num_tokens += token_ids_to_append.size();
    if (committed) {
        num_commit_tokens += token_ids_to_append.size();
    }else{
        num_spec_tokens += token_ids_to_append.size();
    }
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    assert(num_tokens <= block_size);
}

std::vector<TokenId> LogicalTokenBlock::get_token_ids() const {
    return token_ids;
}

PhysicalTokenBlock::PhysicalTokenBlock(int block_number, uint32_t block_size)
    : block_number(block_number), block_size(block_size), ref_count(0) {}

BlockAllocator::BlockAllocator(uint32_t block_size, int num_total_blocks) {
    for (int block_number = 0; block_number < num_total_blocks; ++block_number) {
        free_blocks.push_back(PhysicalTokenBlock(block_number, block_size));
    }
    num_blocks = num_total_blocks;
}

// Allocate a block
PhysicalTokenBlock BlockAllocator::allocate() {
    if (free_blocks.empty()) {
        throw std::runtime_error("Out of memory! No free blocks are available.");
    }
    PhysicalTokenBlock block = free_blocks.front();
    free_blocks.pop_front();
    block.ref_count = 1;
    num_blocks -= 1;
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
        num_blocks += 1;
    }
}

size_t BlockAllocator::get_num_free_blocks() const {
    assert(free_blocks.size() <= static_cast<size_t>(num_blocks));
    if (free_blocks.size() > static_cast<size_t>(num_blocks)) {
        std::cerr << "num free blocks: " << free_blocks.size() << std::endl;
        std::cerr << "num total blocks: " << num_blocks << std::endl;
        throw std::runtime_error("Number of free blocks exceeds the total number of blocks.");
    }
    return free_blocks.size();
}

PageManager::PageManager(uint32_t block_size, int num_total_blocks)
    : block_size(block_size), num_total_blocks(num_total_blocks),
      block_allocator(block_size, num_total_blocks) {}

bool PageManager::prefill(const RequestGuid& request_guid, const std::vector<TokenId>& token_ids) {
    BlockTable block_table;
    for (size_t logical_idx = 0; logical_idx < token_ids.size(); logical_idx++) {
        PhysicalTokenBlock block = block_allocator.allocate();
        block_table.push_back(block);
    }

    block_tables[request_guid] = block_table;
    return true;
}

bool PageManager::can_allocate(const RequestGuid& request_guid) const {
    int num_free_gpu_blocks = block_allocator.get_num_free_blocks();
    return num_free_gpu_blocks > 0;
}

bool PageManager::allocate(const RequestGuid& request_guid) {
    // This is the prefilling for a request
    if (!can_allocate(request_guid)) {
        assert(false);
    }
    BlockTable& block_table = block_tables[request_guid];

    PhysicalTokenBlock block = block_allocator.allocate();
    block_table.push_back(block);;
    return true;
}

void PageManager::_free_block_table(BlockTable& block_table) {
    for (auto& block : block_table) {
            block_allocator.free(block);
    } 
}

void PageManager::free(const RequestGuid& request_guid) {
    assert(block_tables.find(request_guid) != block_tables.end());
    auto& block_table = block_tables[request_guid];
    _free_block_table(block_table);
}

size_t PageManager::get_num_free_blocks() const {
    return block_allocator.get_num_free_blocks();
}

std::vector<int32_t> PageManager::get_block_table_indices(const RequestGuid& request_guid) const {
    std::vector<int32_t> indices;
    try {
    const auto& block_table = block_tables.at(request_guid);
    for (const auto& block : block_table) {
        // printf("get block indice block number is: %d\n", block.block_number);
        indices.push_back(block.block_number);
    }
    } catch (const std::out_of_range& e) {
        std::cerr << "Request GUID not found in block tables: " << e.what() << std::endl;
        // Handle error appropriately
        std::cout << "request ID is: " << request_guid << std::endl;
        exit(1);
    }
    return indices;
}

int PageManager::get_num_allocated_blocks(const RequestGuid& request_guid) const {
    auto it = block_tables.find(request_guid);
    if (it == block_tables.end()) {
        return 0;
    }else{
        return it->second.size();
    }
}

void PageManager::erase_last_pages(const RequestGuid& request_guid, int last_commit_page){
    assert(block_tables.find(request_guid) != block_tables.end());
    auto& block_table = block_tables[request_guid];
    assert(last_commit_page < block_table.size());
    // free the blocks that are used for spec tokens and put them back to the queue
    for (int i = last_commit_page + 1; i < block_table.size(); i++) {
        block_allocator.free(block_table[i]);
    }
    // erase the blocks that are used for spec tokens in the block table of given request
    block_table = std::vector<PhysicalTokenBlock>(block_table.begin(), block_table.begin() + last_commit_page + 1);
    // need to put the last blocks back to the free list
    block_tables[request_guid] = block_table;
    assert(block_tables[request_guid].size() == last_commit_page + 1);
}

PageManager *PageManager::get_page_manager() {
  if (page_manager_singleton == nullptr) {
    int num_total_blocks = (BatchConfig::max_spec_tree_token_num() +
        BatchConfig::max_sequence_length() + kPagesize - 1) /
        kPagesize * BatchConfig::max_requests_per_batch();
    page_manager_singleton = new PageManager(kPagesize, num_total_blocks);
  }
  return page_manager_singleton;
}


}; //FlexFlow