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

RequestManager::RequestManager()
    : next_available_guid(1000000), num_processed_requests(0) {}

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

  std::cout << "new req: " << request.tokens.size() << std::endl;
  return request.guid;
}

size_t RequestManager::get_num_processed_requests() {
  return num_processed_requests;
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
  std::cout << "print all results"
            << "\n";
  for (int i = 0; i < 40; i++) {
    std::cout << result.token_ids[i] << ", ";
  }
  // Step 1: register first batch
  BeamSearchBatchConfig new_bc;
  // Step 2: preparing the next batch for existing requests

  // store results
  std::cout << "depthhhhhhh: " << old_bc.beamRequestsInfo[0].current_depth
            << "\n";
  store_beam_metadata(old_bc, result);

  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (old_bc.request_completed[i]) {
      continue;
    }
    assert(old_bc.requestsInfo[i].num_tokens_in_batch > 0);
    Request &request =
        running_request_queue[old_bc.requestsInfo[i].request_guid];
    int processed_tokens = old_bc.requestsInfo[i].token_start_offset +
                           old_bc.requestsInfo[i].num_tokens_in_batch;

    // std::cout << "processed tokens" << processed_tokens << ", "
    //           << request.tokens.size() << "\n";
    // assert(processed_tokens < request.tokens.size());
    if (processed_tokens >
        old_bc.beamRequestsInfo[i].max_depth + request.tokens.size()
        // || ir.results[t] == 0 TODO: replace this with <EOS>
        // std::cout<<"aaaaaaa"<<"\n";
    ) {
      log_req_mgr.print("[Done] guid(%zu) final_length(%i) request_length(%zu)",
                        old_bc.requestsInfo[i].request_guid,
                        processed_tokens,
                        request.tokens.size());
    } else {

      std::cout << "num tokens: " << old_bc.num_tokens << ", "
                << new_bc.num_tokens;
      new_bc.request_completed[i] = false;
      new_bc.requestsInfo[i].token_start_offset = processed_tokens;
      new_bc.requestsInfo[i].request_guid = old_bc.requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_bc.requestsInfo[i].max_sequence_length;

      // update the beam search metadata
      // how many sub request in current request
      new_bc.sub_requests[i] = old_bc.beamRequestsInfo[i].beam_size;
      // update the parentid, accumalated_probs, depth, and token_ids
      new_bc.beamRequestsInfo[i].current_depth =
          old_bc.beamRequestsInfo[i].current_depth + 1;
      new_bc.beamRequestsInfo[i].beam_size =
          old_bc.beamRequestsInfo[i].beam_size;

      // do the slot exchange to minimize the cache exchange in kernel.
      std::cout << "update metadata" << std::endl;
      update_beam_metadata(new_bc, beam_trees[i], i);

      if (new_bc.requestsInfo[i].token_start_offset + 1 >=
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

      // register more tokens due to the beam width
      for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
        int depth = new_bc.requestsInfo[i].token_start_offset + j;
        for (int k = 0; k < new_bc.sub_requests[i]; k++) {
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;

          // get value from requestinfo
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
              new_bc.beamRequestsInfo[i].tokens[k];
          // request.tokens[depth];
          new_bc.beamTokenInfo[new_bc.num_tokens].sub_request_index = k;
          new_bc.num_tokens++;
        }
      }
    }
  }

  for (int i = 0; i < BeamSearchBatchConfig::MAX_NUM_REQUESTS; i++) {
    if (new_bc.request_completed[i]) {
      if (!pending_request_queue.empty() &&
          new_bc.num_tokens < BeamSearchBatchConfig::MAX_NUM_TOKENS) {
        Request new_request = pending_request_queue.front();
        pending_request_queue.pop();
        running_request_queue[new_request.guid] = new_request;
        new_bc.requestsInfo[i].token_start_offset = 0;
        new_bc.requestsInfo[i].request_guid = new_request.guid;
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(BeamSearchBatchConfig::MAX_NUM_TOKENS - new_bc.num_tokens,
                     (int)new_request.tokens.size());
        new_bc.requestsInfo[i].max_sequence_length =
            new_request.max_sequence_length;

        // init the beam search metadata per request
        new_bc.beamRequestsInfo[i].beam_size =
            BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        new_bc.beamRequestsInfo[i].current_depth = 1;
        for (int j = 0; j < BeamSearchBatchConfig::MAX_BEAM_WIDTH; j++) {
          new_bc.beamRequestsInfo[i].parent_id[j] = 0;
          new_bc.beamRequestsInfo[i].probs[j] = 1;
        }

        new_bc.request_completed[i] = false;
        new_bc.sub_requests[i] = 1;

        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          int depth = new_bc.requestsInfo[i].token_start_offset + j;
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;
          assert(depth < new_request.tokens.size());
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
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

void RequestManager::store_beam_metadata(BeamSearchBatchConfig const &old_bc,
                                         BeamInferenceResult const &result) {
  // step1 store the outputs
  if (old_bc.num_tokens <= 0) {
    return;
  }
  auto guid =
      old_bc.requestsInfo[old_bc.tokensInfo[0].request_index].request_guid;
  auto start_idx = old_bc.tokensInfo[0].abs_depth_in_request;
  int result_index = 0;
  std::cout << "store num tokens" << old_bc.num_tokens << "\n";
  for (int i = 0; i <= old_bc.num_tokens; i++) {
    int request_index = old_bc.tokensInfo[i].request_index;
    if (i == old_bc.num_tokens ||
        old_bc.requestsInfo[request_index].request_guid != guid) {
      // see how many tokens has been put to model in this req
      // to get the index of the final token
      // every token will get (beam_width) results
      int beam_width =
          old_bc.beamRequestsInfo[old_bc.tokensInfo[i].request_index].beam_size;

      result_index +=
          (old_bc.tokensInfo[i - 1].abs_depth_in_request - start_idx) *
          beam_width;

      std::cout << "i = " << i << ", result index = " << result_index
                << ",value: " << result.token_ids[result_index] << "\n";

      int index = old_bc.tokensInfo[i - 1].request_index;
      int beam_size = old_bc.beamRequestsInfo[index].beam_size;
      int depth = old_bc.beamRequestsInfo[index].current_depth;

      if (depth == 1) {
        // store the last input into the tree;
        std::cout << "try to store the input"
                  << "\n";
        Request &request =
            running_request_queue[old_bc.requestsInfo[index].request_guid];
        beam_trees[index].treeLayers[depth - 1].tokens[0] =
            request.tokens.at(request.tokens.size() - 1);
        beam_trees[index].treeLayers[depth - 1].probs[0] = 1;
        beam_trees[index].treeLayers[depth - 1].parent_ids[0] = -1;
        std::cout << "store the previous last token to the tree root"
                  << request.tokens.at(request.tokens.size() - 1) << "\n";
      }

      for (int beam_id = 0; beam_id < beam_width; beam_id++) {
        beam_trees[index].treeLayers[depth].tokens[beam_id] =
            result.token_ids[result_index];
        beam_trees[index].treeLayers[depth].probs[beam_id] =
            result.probs[result_index];
        beam_trees[index].treeLayers[depth].parent_ids[beam_id] =
            result.parent_id[result_index];

        std::cout << "tree value: " << depth << "token: "
                  << beam_trees[index].treeLayers[depth].tokens[beam_id]
                  << "result tokens: " << result.token_ids[result_index];
        result_index += 1;
      }

      if (i < old_bc.num_tokens) {
        guid = old_bc.requestsInfo[request_index].request_guid;
        start_idx = old_bc.tokensInfo[i].abs_depth_in_request;
      }
    }
  }
}

// for updating the beam search metadata in requests in incremental phase
void RequestManager::update_beam_metadata(BeamSearchBatchConfig &new_bc,
                                          BeamTree &tree,
                                          int request_index) {

  // do the exchange
  if (new_bc.request_completed[request_index]) {
    assert(false);
  }
  int depth = new_bc.beamRequestsInfo[request_index].current_depth - 1;
  int beam_size = new_bc.beamRequestsInfo[request_index].beam_size;

  // std::cout << "-----------before parent id exchange-----------" <<
  // std::endl; for (int j = 0; j < beam_size; j++) {
  //   std::cout << "after request id: " << request_index << "beam id = " << j
  //             << "parnt: "
  //             << new_bc.beamRequestsInfo[request_index].parent_id[j]
  //             << "token: " <<
  //             new_bc.beamRequestsInfo[request_index].tokens[j]
  //             << "probs: " << new_bc.beamRequestsInfo[request_index].probs[j]
  //             << std::endl;
  //   // std::fixed << std::setprecision(15)<<
  // }
  if (new_bc.beamRequestsInfo[request_index].current_depth == 1) {
    for (int j = 0; j < beam_size; j++) {
      new_bc.beamRequestsInfo[request_index].parent_id[j] = j;
      new_bc.beamRequestsInfo[request_index].probs[j] =
          tree.treeLayers[depth].probs[j];
      new_bc.beamRequestsInfo[request_index].tokens[j] =
          tree.treeLayers[depth].tokens[j];
    }
  } else {
    std::set<int> parents;
    std::set<int> childs;
    // cache stealing
    for (int j = 0; j < beam_size; j++) {
      int parent_id = tree.treeLayers[depth].parent_ids[j];
      if (childs.find(parent_id) == childs.end()) {
        // copy beam slot
        new_bc.beamRequestsInfo[request_index].parent_id[parent_id] =
            tree.treeLayers[depth].parent_ids[j];
        new_bc.beamRequestsInfo[request_index].probs[parent_id] =
            tree.treeLayers[depth].probs[j];
        new_bc.beamRequestsInfo[request_index].tokens[parent_id] =
            tree.treeLayers[depth].tokens[j];
        parents.emplace(j);
        childs.emplace(parent_id);
      }
    }
    if (parents.size() < beam_size) {
      for (int j = 0; j < beam_size; j++) {
        if (parents.find(j) == parents.end()) {
          // this slot has not been assigned
          // find the smallest not assigned child and put in
          std::cout << "request_index" << request_index << ", miss slot: " << j
                    << "\n";
          for (int k = 0; k < beam_size; k++) {
            if (childs.find(k) == childs.end()) {
              // parent -> j to child k;
              new_bc.beamRequestsInfo[request_index].parent_id[k] =
                  tree.treeLayers[depth].parent_ids[j];
              new_bc.beamRequestsInfo[request_index].probs[k] =
                  tree.treeLayers[depth].probs[j];
              new_bc.beamRequestsInfo[request_index].tokens[k] =
                  tree.treeLayers[depth].tokens[j];
              parents.emplace(j);
              childs.emplace(k);
              break;
            }
          }
        }
      }
    }
  }
  std::cout << "-----------after parent id exchange-----------" << std::endl;
  for (int j = 0; j < beam_size; j++) {
    std::cout << "after request id: " << request_index << "beam id = " << j
              << "parnt: "
              << new_bc.beamRequestsInfo[request_index].parent_id[j]
              << "token: " << new_bc.beamRequestsInfo[request_index].tokens[j]
              << "probs: " << new_bc.beamRequestsInfo[request_index].probs[j]
              << std::endl;
  }
}

bool PreOrder(BeamTree const &tree,
              int max_depth,
              int current_depth,
              int beam_width,
              int id,
              std::vector<std::pair<BeamSearchBatchConfig::TokenId, int>>
                  &serializedTree) {
  // terminate
  if (current_depth >= max_depth) {
    serializedTree.push_back(std::make_pair(
        tree.treeLayers[current_depth].tokens[id], current_depth));
    std::cout << "last tokens: " << tree.treeLayers[current_depth].tokens[id]
              << "\n";
    std::cout << "return true"
              << "\n";
    return true;
  }

  // add to tree;
  // std::cout<<"node: " << current_depth << ", id: " <<
  serializedTree.push_back(
      std::make_pair(tree.treeLayers[current_depth].tokens[id], current_depth));
  std::cout << "push something: " << tree.treeLayers[current_depth].tokens[id]
            << ", " << current_depth << std::endl;
  int index = serializedTree.size() - 1;
  int next_layers = current_depth + 1;

  bool flag = false;
  // recursion
  for (int i = 0; i < beam_width; i++) {
    int child_id = i;
    int child_parent = tree.treeLayers[next_layers].parent_ids[i];

    // for all childs, do preOrder
    if (child_parent == id) {
      std::cout << "current depth: " << current_depth << ", child_parent, "
                << child_parent << ", child_id, " << child_id << "\n";
      bool res = PreOrder(tree,
                          max_depth,
                          current_depth + 1,
                          beam_width,
                          child_id,
                          serializedTree);
      flag = flag || res;
    }
  }
  if (!flag) {
    // no child for this token, delete it
    std::cout << "delete a node: " << tree.treeLayers[current_depth].tokens[id]
              << ", " << current_depth << std::endl;
    serializedTree.erase(serializedTree.begin() + index);
  }
  return flag;
}

TreeVerifyBatchConfig RequestManager::convert_beam_to_tree_batch_config(
    BeamSearchBatchConfig const &beam_bc) {
  TreeVerifyBatchConfig tree_bc;
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (beam_bc.request_completed[i]) {
      continue;
    }
    // We don't modify requests during the conversion
    tree_bc.request_completed[i] = beam_bc.request_completed[i];
    BeamTree const &tree = beam_trees[i];
    // token, index
    // todo make this one global for different stages
    std::vector<std::pair<BeamSearchBatchConfig::TokenId, int>> serializedTree;
    PreOrder(tree,
             beam_bc.beamRequestsInfo[i].max_depth,
             0,
             beam_bc.beamRequestsInfo[i].beam_size,
             0,
             serializedTree);
    tree_bc.requestsInfo[i].request_guid = beam_bc.requestsInfo[i].request_guid;
    tree_bc.requestsInfo[i].max_sequence_length =
        beam_bc.requestsInfo[i].max_sequence_length;
    tree_bc.requestsInfo[i].token_start_offset = serializedTree[0].second;
    tree_bc.requestsInfo[i].num_tokens_in_batch = 0;
    for (int k = 0; k < serializedTree.size(); k++) {
      assert(tree_bc.num_tokens < BatchConfig::MAX_NUM_TOKENS);
      tree_bc.tokensInfo[tree_bc.num_tokens].request_index = i;
      tree_bc.tokensInfo[tree_bc.num_tokens].abs_depth_in_request =
          serializedTree[k].second;
      tree_bc.tokensInfo[tree_bc.num_tokens].token_id = serializedTree[k].first;
      tree_bc.num_tokens++;
      tree_bc.requestsInfo[i].num_tokens_in_batch++;
    }
  }
  return tree_bc;
}

void RequestManager::tranverse_beam_tree(BeamSearchBatchConfig const &old_bc) {
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (old_bc.request_completed[i]) {
      continue;
    }
    // if(i != 0){
    //   continue;
    // }

    int depth = old_bc.beamRequestsInfo[i].current_depth;
    int beam_width = old_bc.beamRequestsInfo[i].beam_size;
    BeamTree const &tree = beam_trees[i];

    // token, index
    // todo make this one global for different stages
    std::vector<std::pair<BeamSearchBatchConfig::TokenId, int>> serializedTree;
    PreOrder(
        tree, 3, 0, old_bc.beamRequestsInfo[i].beam_size, 0, serializedTree);

    // print it
    std::cout << "print tree, " << i << "\n";
    for (int k = 0; k < serializedTree.size(); k++) {
      std::cout << "token id: " << serializedTree.at(k).first
                << ", depth: " << serializedTree.at(k).second << "\n";
    }
  }
}

}; // namespace FlexFlow
