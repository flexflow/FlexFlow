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

#include "flexflow/inference.h"
#include "flexflow/parallel_ops/parallel_op.h"

namespace FlexFlow {

using namespace Legion;

LegionRuntime::Logger::Category log_req_mgr("RequestManager");

RequestManager::RequestManager() : next_available_guid(1000000) {}

RequestManager::RequestGuid
    RequestManager::register_new_request(std::vector<TokenId> const &prompt,
                                         int max_sequence_length) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);

  // Add a new request
  Request request;
  request.guid = next_available_guid++;
  request.max_sequence_length = max_sequence_length;
  request.tokens = prompt;

  pending_request_queue.push(request);
  return request.guid;
}

BatchConfig RequestManager::prepare_next_batch(BatchConfig const &old_bc,
                                               InferenceResult const &result) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  // Step 1: use result to update requests
  for (int i = 0; i < old_bc.num_tokens; i++) {
    size_t guid =
        old_bc.requestsInfo[old_bc.tokensInfo[i].request_index].request_guid;
    Request &request = running_request_queue[guid];
    if (old_bc.tokensInfo[i].abs_depth_in_request + 1 < request.tokens.size()) {
      // This is a prompt token
      continue;
    } else {
      assert(old_bc.tokensInfo[i].abs_depth_in_request + 1 ==
             request.tokens.size());
      // This is a decoding token
      request.tokens.push_back(result.token_ids[i]);
    }
  }
  // Step 2: preparing the next batch for existing requests
  BatchConfig new_bc;
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (old_bc.request_completed[i]) {
      continue;
    }
    assert(old_bc.requestsInfo[i].num_tokens_in_batch > 0);
    Request &request =
        running_request_queue[old_bc.requestsInfo[i].request_guid];
    int processed_tokens = old_bc.requestsInfo[i].token_start_offset +
                           old_bc.requestsInfo[i].num_tokens_in_batch;
    assert(processed_tokens < request.tokens.size());
    if (request.tokens.size() >= old_bc.requestsInfo[i].max_sequence_length
        // || ir.results[t] == 0 TODO: replace this with <EOS>
    ) {
      log_req_mgr.print("[Done] guid(%zu) final_length(%zu)",
                        old_bc.requestsInfo[i].request_guid,
                        request.tokens.size());
    } else {
      new_bc.request_completed[i] = false;
      new_bc.requestsInfo[i].token_start_offset = processed_tokens;
      new_bc.requestsInfo[i].request_guid = old_bc.requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_bc.requestsInfo[i].max_sequence_length;
      if (new_bc.requestsInfo[i].token_start_offset + 1 ==
          request.tokens.size()) {
        // Incremental phase
        new_bc.requestsInfo[i].num_tokens_in_batch = 1;
      } else {
        // Prompt phase
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(BatchConfig::MAX_NUM_TOKENS - new_bc.num_tokens,
                     (int)request.tokens.size() -
                         new_bc.requestsInfo[i].token_start_offset);
      }
      for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
        int depth = new_bc.requestsInfo[i].token_start_offset + j;
        new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
        new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;
        assert(depth < request.tokens.size());
        new_bc.tokensInfo[new_bc.num_tokens].token_id = request.tokens[depth];
        new_bc.num_tokens++;
      }
    }
  }
  // Step 3: add new requests to the next batch
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (new_bc.request_completed[i]) {
      if (!pending_request_queue.empty() &&
          new_bc.num_tokens < BatchConfig::MAX_NUM_TOKENS) {
        Request new_request = pending_request_queue.front();
        pending_request_queue.pop();
        running_request_queue[new_request.guid] = new_request;
        new_bc.requestsInfo[i].token_start_offset = 0;
        new_bc.requestsInfo[i].request_guid = new_request.guid;
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(BatchConfig::MAX_NUM_TOKENS - new_bc.num_tokens,
                     (int)new_request.tokens.size());
        new_bc.requestsInfo[i].max_sequence_length =
            new_request.max_sequence_length;
        new_bc.request_completed[i] = false;
        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          int depth = new_bc.requestsInfo[i].token_start_offset + j;
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;
          assert(depth < new_request.tokens.size());
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
              new_request.tokens[depth];
          new_bc.num_tokens++;
        }
        if (new_bc.num_tokens == BatchConfig::MAX_NUM_TOKENS) {
          break;
        }
      }
    }
  }
  return new_bc;
}

//-------beam search specific functions

// update beam search metadata
BeamSearchBatchConfig
    RequestManager::prepare_next_batch_beam(BeamSearchBatchConfig const &old_bc,
                                            BeamInferenceResult const &result) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  // Step 1: use result to update requests
  for (int i = 0; i < old_bc.num_tokens; i++) {
    size_t guid = old_bc.beamRequestsInfo[old_bc.tokensInfo[i].request_index]
                      .request_guid;
    Request &request = running_request_queue[guid];
    if (old_bc.tokensInfo[i].abs_depth_in_request + 1 < request.tokens.size()) {
      // This is a prompt token
      continue;
    } else {
      assert(old_bc.tokensInfo[i].abs_depth_in_request + 1 ==
             request.tokens.size());
      // This is a decoding token
      request.tokens.push_back(result.token_ids[i]);
      // for(int j = 0; j < old_bc.beamRequestsInfo[i].beam_size; i++){
      //     request.tokens.push_back(result.token_ids[i + j]);
      // }
    }
  }

  // Step 2: preparing the next batch for existing requests
  BeamSearchBatchConfig new_bc;
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (old_bc.request_completed[i]) {
      continue;
    }
    assert(old_bc.beamRequestsInfo[i].num_tokens_in_batch > 0);
    Request &request =
        running_request_queue[old_bc.beamRequestsInfo[i].request_guid];
    int processed_tokens = old_bc.beamRequestsInfo[i].token_start_offset +
                           old_bc.beamRequestsInfo[i].num_tokens_in_batch;
    assert(processed_tokens < request.tokens.size());
    if (processed_tokens >= old_bc.beamRequestsInfo[i].max_sequence_length
        // || ir.results[t] == 0 TODO: replace this with <EOS>
    ) {
      log_req_mgr.print("[Done] guid(%zu) final_length(%zu)",
                        old_bc.beamRequestsInfo[i].request_guid,
                        request.tokens.size());
    } else {

      new_bc.request_completed[i] = false;
      new_bc.beamRequestsInfo[i].token_start_offset = processed_tokens;
      new_bc.beamRequestsInfo[i].request_guid =
          old_bc.beamRequestsInfo[i].request_guid;
      new_bc.beamRequestsInfo[i].max_sequence_length =
          old_bc.beamRequestsInfo[i].max_sequence_length;

      // update the beam search metadata
      // how many sub request in current request
      new_bc.sub_requests[i] = old_bc.beamRequestsInfo[i].beam_size;
      // update the parentid, accumalated_probs, depth, and token_ids
      new_bc.beamRequestsInfo[i].current_depth =
          old_bc.beamRequestsInfo[i].current_depth + 1;


      //do the slot exchange to minimize the cache exchange in kernel.
      update_beam_metadata(new_bc, result);
      
      if (new_bc.beamRequestsInfo[i].token_start_offset + 1 ==
          request.tokens.size()) {
        // Incremental phase
        new_bc.beamRequestsInfo[i].num_tokens_in_batch = 1;
      } else {
        // Prompt phase
        new_bc.beamRequestsInfo[i].num_tokens_in_batch =
            std::min(BatchConfig::MAX_NUM_TOKENS - new_bc.num_tokens,
                     (int)request.tokens.size() -
                         new_bc.beamRequestsInfo[i].token_start_offset);
      }

      // register more tokens due to the beam width
      for (int j = 0; j < new_bc.beamRequestsInfo[i].num_tokens_in_batch; j++) {
        int depth = new_bc.beamRequestsInfo[i].token_start_offset + j;
        for (int k = 0; k < new_bc.sub_requests[i]; k++) {
          new_bc.beamTokenInfo[new_bc.num_tokens].request_index = i;
          new_bc.beamTokenInfo[new_bc.num_tokens].abs_depth_in_request = depth;
          assert(depth < request.tokens.size());

          //get value from requestinfo
          new_bc.beamTokenInfo[new_bc.num_tokens].token_id = new_bc.beamRequestsInfo[i].tokens[j];
              // request.tokens[depth];
          new_bc.beamTokenInfo[new_bc.num_tokens].sub_request_index = k;
          new_bc.num_tokens++;
        }
      }
    }
  }
  // Step 3: add new requests to the next batch
  for (int i = 0; i < BeamSearchBatchConfig::MAX_NUM_REQUESTS; i++) {
    if (new_bc.request_completed[i]) {
      if (!pending_request_queue.empty() &&
          new_bc.num_tokens < BeamSearchBatchConfig::MAX_NUM_TOKENS) {
        Request new_request = pending_request_queue.front();
        pending_request_queue.pop();
        running_request_queue[new_request.guid] = new_request;
        new_bc.beamRequestsInfo[i].token_start_offset = 0;
        new_bc.beamRequestsInfo[i].request_guid = new_request.guid;
        new_bc.beamRequestsInfo[i].num_tokens_in_batch =
            std::min(BeamSearchBatchConfig::MAX_NUM_TOKENS - new_bc.num_tokens,
                     (int)new_request.tokens.size());
        new_bc.beamRequestsInfo[i].max_sequence_length =
            new_request.max_sequence_length;

        // init the beam search metadata per request
        new_bc.beamRequestsInfo[i].beam_size =
            BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        new_bc.beamRequestsInfo[i].current_depth = 0;
        for (int j = 0; j < BeamSearchBatchConfig::MAX_BEAM_WIDTH; j++) {
          new_bc.beamRequestsInfo[i].parent_id[j] = 0;
          new_bc.beamRequestsInfo[i].probs[j] = 1;
        }

        new_bc.request_completed[i] = false;
        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          int depth = new_bc.requestsInfo[i].token_start_offset + j;
          new_bc.beamTokenInfo[new_bc.num_tokens].request_index = i;
          new_bc.beamTokenInfo[new_bc.num_tokens].abs_depth_in_request = depth;
          assert(depth < new_request.tokens.size());
          new_bc.beamTokenInfo[new_bc.num_tokens].token_id =
              new_request.tokens[depth];

          // beam search meta data, indicate which sub request this token
          // belongs to, init to 0;
          new_bc.beamTokenInfo[new_bc.num_tokens].sub_request_index = 0;
          new_bc.num_tokens++;
        }
        if (new_bc.num_tokens == BatchConfig::MAX_NUM_TOKENS) {
          break;
        }
      }
    }
  }
  return new_bc;
}

//for updating the beam search metadata in requests in incremental phase
void RequestManager::update_beam_metadata(BeamSearchBatchConfig bc,
                                     BeamInferenceResult const &result) {

  // step1 store the outputs
  auto guid = bc.beamRequestsInfo[bc.tokensInfo[0].request_index].request_guid;
  auto start_idx = bc.beamTokenInfo[0].abs_depth_in_request;
  int result_index = 0;
  for (int i = 0; i <= bc.num_tokens; i++) {
    int request_index = bc.beamTokenInfo[i].request_index;
    if (i == bc.num_tokens ||
        bc.beamRequestsInfo[request_index].request_guid != guid) {
      // see how many tokens has been put to model in this req
      // to get the index of the final token

      // every token will get (beam_width) results
      int beam_width =
          bc.beamRequestsInfo[bc.beamTokenInfo[i].request_index].beam_size;

      result_index +=
          (bc.beamTokenInfo[i - 1].abs_depth_in_request - start_idx) *
          beam_width;

      for (int beam_id = 0; beam_id < beam_width; beam_id++) {
        bc.beamRequestsInfo[request_index].tokens[beam_id] =
            result.token_ids[result_index];
        bc.beamRequestsInfo[request_index].probs[beam_id] =
            result.probs[result_index];
        bc.beamRequestsInfo[request_index].parent_id[beam_id] =
            result.parent_id[result_index];
        result_index += 1;
      }

      if (i < bc.num_tokens) {
        guid = bc.beamRequestsInfo[request_index].request_guid;
        start_idx = bc.beamTokenInfo[i].abs_depth_in_request;
      }
    }
  }

  // step2 do the exchange
  for (int i = 0; i < bc.MAX_NUM_REQUESTS; i++) {
    if (bc.request_completed[i]) {
      continue;
    }

    int beam_size = bc.beamRequestsInfo[i].beam_size;
    if (bc.beamRequestsInfo[i].current_depth == 1) {
      for (int j = 0; j < beam_size; j++) {
        bc.beamRequestsInfo[i].parent_id[j] = j;
        bc.beamRequestsInfo[i].probs[j] =  bc.beamRequestsInfo[i].probs[j];
        bc.beamRequestsInfo[i].tokens[j] =  bc.beamRequestsInfo[i].tokens[j];
      }
    } else {
      std::set<int> parents;
      std::set<int> childs;
      // cache stealing
      for (int j = 0; j < beam_size; j++) {
        int parent_id = bc.beamRequestsInfo[i].parent_id[j];
        if (childs.find(parent_id) == childs.end()) {
          // copy beam slot
          //do nothing
          // bc->beam_slots.at(i).parent_id[parent_id] = result.parent_ids[j];
          // bc->beam_slots.at(i).probs[parent_id] = result.probs[j];
          // bc->beam_slots.at(i).tokens[parent_id] = result.tokens[j];
          parents.emplace(j);
          childs.emplace(parent_id);
        }
      }
      if (parents.size() < beam_size) {
        for (int j = 0; j < beam_size; j++) {
          if (parents.find(j) == parents.end()) {
            // this slot has not been assigned
            // find the smallest not assigned child and put in
            for (int k = 0; k < beam_size; k++) {
              if (childs.find(k) == childs.end()) {
                // parent -> j to child k;
                int temp_parent_id =  bc.beamRequestsInfo[i].parent_id[j];
                float temp_probs = bc.beamRequestsInfo[i].probs[j];
                BatchConfig::TokenId temp_token_id =  bc.beamRequestsInfo[i].tokens[j];
                bc.beamRequestsInfo[i].parent_id[k] = temp_parent_id;
                bc.beamRequestsInfo[i].probs[k] = temp_probs;
                bc.beamRequestsInfo[i].tokens[k] = temp_token_id;
                parents.emplace(j);
                childs.emplace(k);
                break;
              }
            }
          }
        }
      }
    }
    // std::cout << "-----------after parent id exchange-----------" << std::endl;
    // for (int j = 0; j < beam_size; j++) {
    //   std::cout << "after request id: " << i << "beam id = " << j
    //             << "parnt: " << bc.beam_slots[i].parent_id[j]
    //             << "token: " << bc.beam_slots[i].tokens[j]
    //             << "probs: " << bc.beam_slots[i].probs[j] << std::endl;
      // std::fixed << std::setprecision(15)<<
    // }
  }
}

}; // namespace FlexFlow
