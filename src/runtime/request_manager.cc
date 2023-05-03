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

SpeculativeRequestManager::SpeculativeRequestManager() : next_available_guid(1000000) {}

SpeculativeRequestManager::~SpeculativeRequestManager() {}

BeamSearchBatchConfig 
  SpeculativeRequestManager::prepare_next_batch_for_speculation(BeamSearchBatchConfig const &old_bc, 
                                                                InferenceResult const &result)
{
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  // Step 1: use result to update requests
  for (int i = 0; i < old_bc.num_tokens; i++) {
    size_t guid =
        old_bc.requestsInfo[old_bc.tokensInfo[i].request_index].request_guid;
    SpecRequest &request = running_request_queue[guid];
    if (old_bc.tokensInfo[i].abs_depth_in_request + 1 < request.tokens.size() + bc.iterations) {
      // This is a prompt token
      continue;
    } else {
      assert(old_bc.tokensInfo[i].abs_depth_in_request + 1 ==
             request.tokens.size());
      // This is a decoding token
      request.candidate_tokens.push_back(result.token_ids[i]);
      request.candidate_parent_ids.push_back(result.parent_ids[i]);
      request.candidate_cum_probs.push_back(result.cum_probs[i]);
    }
  }
  
  // Step 2: preparing the next batch for existing requests
  BeamSearchBatchConfig new_bc;
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (old_bc.request_completed[i]) {
      continue;
    }
    assert(old_bc.requestsInfo[i].num_tokens_in_batch > 0);
    SpecRequest &request =
        running_request_queue[old_bc.requestsInfo[i].request_guid];
    int processed_tokens = old_bc.requestsInfo[i].token_start_offset +
                           old_bc.requestsInfo[i].num_tokens_in_batch;
    // assert(processed_tokens < request.tokens.size());
    if (old_bc.done()) { // Finalize the request for verification
      log_req_mgr.print("[Done] guid(%zu) with spec_tree_depth(%d)",
                        old_bc.requestsInfo[i].request_guid,
                        bc.iterations);
    } else {
      // Update new bc's request info, each sub request take one spot
      new_bc.request_completed[i] = false;
      new_bc.requestsInfo[i].token_start_offset = processed_tokens;
      new_bc.requestsInfo[i].request_guid = old_bc.requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_bc.requestsInfo[i].max_sequence_length;
      new_bc.iterations = old_bc.iterations + 1;

      if (new_bc.requestsInfo[i].token_start_offset + 1 ==
          request.tokens.size() + new_bc.iterations) {
        // Incremental phase
        new_bc.requestsInfo[i].num_tokens_in_batch = 1;
      } else {
        // Prompt phase (check this part later)
        assert(new_bc.iterations == 0); // Prompt phase only happens in the first batch
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(BatchConfig::MAX_NUM_TOKENS - new_bc.num_tokens,
                     (int)request.tokens.size() -
                         new_bc.requestsInfo[i].token_start_offset);
      }

      // Update new bc's token info
      for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
        int depth = new_bc.requestsInfo[i].token_start_offset + j;
        new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
        new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;
        assert(depth < request.tokens.size() + new_bc.iterations);
        new_bc.tokensInfo[new_bc.num_tokens].token_id = request.tokens[depth];
        new_bc.num_tokens++;
      }
    }
  }

  // Step 3: add new requests to the next batch (should not happens in beam searchspeculation phase)

  if(old_bc.done()) { // check this logic later
    return std::nullptr;
  }

  return new_bc;
}

}; // namespace FlexFlow
