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

PageManager *page_manager_singleton = nullptr;

LogicalTokenBlock::LogicalTokenBlock(int block_number, uint32_t block_size)
    : block_number(block_number), block_size(block_size), num_tokens(0) {
        // WARNING: init token lists as empty vector
    }

bool LogicalTokenBlock::is_empty() const {
    return num_tokens == 0;
}

int LogicalTokenBlock::get_num_empty_slots() const {
    return block_size - num_tokens;
}

int LogicalTokenBlock::get_num_alloc_slots() {
    return num_tokens;
}

bool LogicalTokenBlock::is_full() const {
    return num_tokens == block_size;
}

void LogicalTokenBlock::reset_num_spec_tokens(){
    assert(num_spec_tokens + num_commit_tokens == num_tokens);
    num_tokens -= num_spec_tokens;
    num_spec_tokens = 0;
}

void LogicalTokenBlock::append_tokens(const std::vector<TokenId>& token_ids_to_append, bool committed) {
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
}

std::vector<int> LogicalTokenBlock::get_token_ids() const {
    return token_ids;
}

int LogicalTokenBlock::get_last_token_id() const {
    if (num_tokens == 0) {
        throw std::runtime_error("Block is empty! Cannot get last token id.");
    }
    return token_ids.back();
}

PhysicalTokenBlock::PhysicalTokenBlock(int block_number, uint32_t block_size)
    : block_number(block_number), block_size(block_size), ref_count(0) {}

BlockAllocator::BlockAllocator(uint32_t block_size, int num_total_blocks) {
    for (int block_number = 0; block_number < num_total_blocks; ++block_number) {
        free_blocks.push_back(PhysicalTokenBlock(block_number, block_size));
    }
}

// Allocate a block
PhysicalTokenBlock BlockAllocator::allocate() {
    if (free_blocks.empty()) {
        throw std::runtime_error("Out of memory! No free blocks are available.");
    }
    PhysicalTokenBlock block = free_blocks.front();
    free_blocks.pop_front();
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

PageManager::PageManager(uint32_t block_size, int num_total_blocks)
    : block_size(block_size), num_total_blocks(num_total_blocks),
      gpu_allocator(block_size, num_total_blocks) {}

bool PageManager::can_prefill(const RequestGuid& request_guid, const std::vector<int>& token_ids) {
    // check how many blocks are needed
    int num_blocks_needed = std::ceil(token_ids.size() / block_size);
    return num_blocks_needed <= gpu_allocator.get_num_free_blocks();
}

// initalize for blocks
bool PageManager::prefill(const RequestGuid& request_guid, const std::vector<int>& token_ids) {
    // This is the prefilling for a request
    if (!can_prefill(request_guid, token_ids)) {
        std::cout << "Cannot prefill for request " << request_guid << std::endl;
        return false;
    }
    BlockTable block_table;
    for (size_t logical_idx = 0; logical_idx < token_ids.size(); ++logical_idx) {
        PhysicalTokenBlock block = gpu_allocator.allocate();
        block_table.push_back(block);
    }

    block_tables[request_guid] = block_table;
    return true;
}

//TODO: check these functions later
bool PageManager::can_allocate(const RequestGuid& request_guid) const {
    int num_free_gpu_blocks = gpu_allocator.get_num_free_blocks();
    return num_free_gpu_blocks > 0;
}

bool PageManager::allocate(const RequestGuid& request_guid) {
    // This is the prefilling for a request
    if (!can_allocate(request_guid)) {
        return false;
    }
    BlockTable& block_table = block_tables[request_guid];

    PhysicalTokenBlock block = gpu_allocator.allocate();
    block_table.push_back(block);
    printf("allocated block %d\n", block.block_number);
    return true;
}


void PageManager::_free_block_table(BlockTable& block_table) {
    for (auto& block : block_table) {
            gpu_allocator.free(block);
    } 
}

void PageManager::free(const RequestGuid& request_guid) {
    auto& block_table = block_tables[request_guid];
    _free_block_table(block_table);
}

int PageManager::get_num_free_blocks() const {
    return gpu_allocator.get_num_free_blocks();
}

std::vector<int> PageManager::get_block_table_indices(const RequestGuid& request_guid) const {
    std::vector<int> indices;
    const auto& block_table = block_tables.at(request_guid);
    for (const auto& block : block_table) {
        indices.push_back(block.block_number);
    }
    return indices;
}

int PageManager::get_num_allocated_blocks(const RequestGuid& request_guid) const {
    // printf("called num allocated blocks\n");
    auto it = block_tables.find(request_guid);
    if (it == block_tables.end()) {
        return 0;
    }else{
        return it->second.size();
    }
}

void PageManager::erase_last_pages(const RequestGuid& request_guid, int num_pages){
    auto& block_table = block_tables[request_guid];
    assert(num_pages <= block_table.size());
    // erase the last num_pages blocks
    block_table.erase(block_table.end() - num_pages, block_table.end());
    block_tables[request_guid] = block_table;
}

int PageManager::lookup_index(const RequestGuid& request_guid, int logical_index){
    auto& block_table = block_tables[request_guid];
    int block_index = (logical_index + block_size - 1) / block_size;
    int block_offset = logical_index % block_size;
    return block_table[block_index].block_number * block_size + block_offset;
}

PageManager *PageManager::get_page_manager() {
  if (page_manager_singleton == nullptr) {
    // FIXME: These values are hardcoded for now
    page_manager_singleton = new PageManager(kPagesize, (BatchConfig::max_spec_tree_token_num() +
        BatchConfig::max_sequence_length() + kPagesize - 1) /
        kPagesize * BatchConfig::max_requests_per_batch());
  }
  return page_manager_singleton;
}


}; //FlexFlow