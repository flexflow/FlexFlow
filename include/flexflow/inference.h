/* Copyright 2022 CMU, Stanford, Facebook, LANL
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
#include "flexflow/model.h"
#include <mutex>

namespace FlexFlow {

class FFModel;
class BeamTree;

class InferenceManager {
public:
  InferenceManager(FFConfig const &config,
                   int max_num_tokens_per_batch,
                   int max_num_inflight_batches);
  void compile_model_and_allocate_buffer(
      FFModel *model,
      std::unordered_map<Tensor, std::vector<MachineView>> const &mapping);
  void init_operators_inference(FFModel *model);
  MachineView *get_machine_view(int mv_id);
  Legion::FutureMap inference(FFModel *model, int index, BatchConfig const &bc);
  void load_input_tokens_from_batch_config(BatchConfig const &bc,
                                           ParallelTensor const input);
  void load_positions(BatchConfig const &bc, ParallelTensor position_input);

public:
  FFConfig ff_config;
  std::unordered_map<ParallelTensor, std::vector<ParallelTensor>> tensor_buffer;
  int max_num_tokens_per_batch;
  int max_num_inflight_batches;
  int num_devices;
  std::vector<MachineView> machine_views;
};

struct Request {
  BatchConfig::RequestGuid guid;
  int max_sequence_length;
  int initial_len;
  std::vector<BatchConfig::TokenId> tokens;
};

// store the result of beam search
struct BeamTree {
  struct treeLayer {
    BeamSearchBatchConfig::TokenId
        tokens[BeamSearchBatchConfig::MAX_BEAM_WIDTH];
    int parent_ids[BeamSearchBatchConfig::MAX_BEAM_WIDTH];
    float probs[BeamSearchBatchConfig::MAX_BEAM_WIDTH];
  };
  treeLayer treeLayers[BeamSearchBatchConfig::MAX_BEAM_DEPTH + 1];
};

// struct BeamTree_v2 {
//   std::vector<BatchConfig::TokenId> tokens;
//   std::vector<int> parent_ids;
//   std::vector<float> probs;
// };

class Tokenizer;

class RequestManager {
public:
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;
  RequestManager(Tokenizer *tokenizer,
                 bool verbose = false,
                 std::string output_filepath = "");
  RequestManager();
  size_t get_num_processed_requests();
  RequestGuid register_new_request(std::string const &prompt,
                                   int max_sequence_length);
  RequestGuid register_new_request(std::vector<TokenId> const &prompt,
                                   int max_sequence_length);
  BatchConfig prepare_next_batch(BatchConfig const &bc,
                                 InferenceResult const &result);
  BeamSearchBatchConfig
      prepare_next_batch_beam(BeamSearchBatchConfig const &old_bc,
                              BeamInferenceResult const &result);

  BeamSearchBatchConfig
      prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                              InferenceResult const &result);

  TreeVerifyBatchConfig
      prepare_next_batch_verify(BeamSearchBatchConfig const &old_bc);

  void store_beam_metadata(BeamSearchBatchConfig const &old_bc,
                           BeamInferenceResult const &result);
  void update_beam_metadata(BeamSearchBatchConfig &new_bc,
                            BeamTree &tree,
                            int request_index);

  std::vector<std::pair<BatchConfig::TokenId, int>>
      traverse_beam_tree(BeamSearchBatchConfig const &old_bc,
                         int request_index,
                         int token_start_offset);

  std::vector<std::pair<BatchConfig::TokenId, int>> traverse_verify_tree(
      size_t guid,
      std::vector<std::pair<BatchConfig::TokenId, int>> const
          &inputSerializedTree,
      std::vector<std::pair<BatchConfig::TokenId, int>> const
          &outputSerializedTree);
  int get_requests_init_length(BeamSearchBatchConfig const &old_bc);      

  // TreeVerifyBatchConfig
  //     convert_beam_to_tree_batch_config(BeamSearchBatchConfig const
  //     &beam_bc);

  static void
      load_tokens_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      load_positions_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);

private:
  Tokenizer *tokenizer;
  bool verbose;
  std::string output_filepath;
  std::queue<Request> pending_request_queue;
  std::unordered_map<RequestGuid, Request> running_request_queue;
  std::mutex request_queue_mutex;
  RequestGuid next_available_guid;

  struct BeamTree beam_trees[BatchConfig::MAX_NUM_REQUESTS];

  std::unordered_map<RequestGuid,
                     std::vector<std::pair<BatchConfig::TokenId, int>>>
      dfs_tree_inputs_map;

  // std::unordered_map<RequestGuid, BeamTree_v2> beam_trees_v2;
  // TODO: cache config info for Verify/Beam exchange: Beam Width, Beam Depth,
  // Committed Tokens
  std::unordered_map<RequestGuid, std::vector<std::pair<int, int>>>
      committed_tokens;
  // Performance profiling
  size_t num_processed_requests;

private:
  struct ProfileInfo {
    int decoding_steps;
    double start_time, finish_time;
  };
  std::unordered_map<RequestGuid, ProfileInfo> profiling_requests;
  double total_request_run_time;
};

} // namespace FlexFlow
