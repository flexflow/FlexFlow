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
#include <mutex>
#include <tokenizers_cpp.h>

namespace FlexFlow {

class FFModel;
class BeamTree;
class RequestManager;
using tokenizers::Tokenizer;

class InferenceManager {
public:
  InferenceManager(FFConfig const &config, int max_num_tokens_per_batch);
  static InferenceManager *get_inference_manager();
  void compile_model_and_allocate_buffer(FFModel *model);
  void init_operators_inference(FFModel *model);
  Legion::FutureMap inference(FFModel *model, int index, BatchConfig const &bc);
  Legion::FutureMap
      inference(FFModel *model, int index, BatchConfigFuture const &bc);
  void load_input_tokens_from_batch_config(BatchConfigFuture const &bc,
                                           ParallelTensor const input);
  void load_positions(BatchConfigFuture const &bc,
                      ParallelTensor position_input,
                      int offset);

public:
  FFConfig ff_config;
  std::unordered_map<ParallelTensor, std::vector<ParallelTensor>> tensor_buffer;
  int max_num_tokens_per_batch;
  int num_devices;
};

struct Request {
  enum Status {
    PENDING = 101,   // loading prompt
    RUNNING = 102,   // running inference
    COMPLETED = 103, // finished and verified
    FINISHING = 104, // finishing request, but not yet verified
  };
  BatchConfig::RequestGuid guid;
  int max_sequence_length;
  int initial_len;
  int ssm_cache_size = 0;
  int llm_cache_size = 0;

  Status status = PENDING;
  std::vector<BatchConfig::TokenId> tokens;

  std::vector<struct BeamTree> beam_trees;
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

class RequestManager {
public:
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;

  RequestManager();
  static RequestManager *get_request_manager();
  size_t get_num_processed_requests();
  size_t get_num_ssms();

  int register_ssm_model(FFModel *model);
  void register_tokenizer(ModelType model_type,
                          int bos_token_id,
                          int eos_token_id,
                          std::string const &path);
  void register_output_filepath(std::string const &);

  FFModel *get_model(int model_id);

  GenerationResult generate_incr_decoding(FFModel *model,
                                          std::vector<std::string> &prompts,
                                          int max_seq_length);
  GenerationResult generate_spec_infer(FFModel *model,
                                       std::vector<std::string> &prompts,
                                       int max_seq_length);
  GenerationResult get_generation_result(RequestGuid const &guid);
  RequestGuid register_new_request(std::string const &prompt,
                                   int max_sequence_length);
  RequestGuid register_new_request(std::vector<TokenId> const &prompt,
                                   int max_sequence_length);
  bool is_request_completed(RequestGuid const &guid);
  BatchConfig prepare_next_batch(BatchConfig const &bc,
                                 InferenceResult const &result);
  BatchConfigFuture prepare_next_batch(BatchConfigFuture const &bc,
                                       InferenceResultFuture const &result);
  BeamSearchBatchConfig
      prepare_next_batch_beam(BeamSearchBatchConfig const &old_bc,
                              BeamInferenceResult const &result);
  BeamSearchBatchConfigFuture
      prepare_next_batch_beam(BeamSearchBatchConfigFuture const &old_bc,
                              BeamInferenceResultFuture const &result);
  BeamSearchBatchConfig
      prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                              InferenceResult const &result,
                              int model_id);
  BeamSearchBatchConfigFuture
      prepare_next_batch_init(TreeVerifyBatchConfigFuture const &old_bc,
                              InferenceResultFuture const &result,
                              int model_id);
  TreeVerifyBatchConfig prepare_next_batch_verify(
      std::vector<BeamSearchBatchConfig> const &old_batches);
  TreeVerifyBatchConfigFuture prepare_next_batch_verify(
      std::vector<BeamSearchBatchConfigFuture> const &old_batches);

  void store_beam_metadata(BeamSearchBatchConfig const &old_bc,
                           BeamInferenceResult const &result);
  void update_beam_metadata(BeamSearchBatchConfig &new_bc,
                            BeamTree &tree,
                            int request_index);

  std::vector<std::pair<BatchConfig::TokenId, int>>
      traverse_beam_tree(BeamSearchBatchConfig const &old_bc,
                         int request_index,
                         int token_start_offset);

  // remove guid after put the cached tree in request
  std::vector<std::pair<BatchConfig::TokenId, int>> merge_dfs_trees(
      std::vector<std::vector<std::pair<BatchConfig::TokenId, int>>>
          input_trees,
      int root_depth,
      RequestGuid guid);

  std::vector<std::pair<BatchConfig::TokenId, int>> traverse_verify_tree(
      size_t guid,
      std::vector<std::pair<BatchConfig::TokenId, int>> const
          &inputSerializedTree,
      std::vector<std::pair<BatchConfig::TokenId, int>> const
          &outputSerializedTree);

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

  static BatchConfig prepare_next_batch_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static BeamSearchBatchConfig prepare_next_batch_beam_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static BeamSearchBatchConfig prepare_next_batch_init_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static TreeVerifyBatchConfig prepare_next_batch_verify_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

private:
  std::unique_ptr<Tokenizer> tokenizer_;
  bool verbose;
  ModelType model_type;
  int bos_token_id;
  int eos_token_id;
  std::string output_filepath;
  std::queue<Request> pending_request_queue;
  std::unordered_map<RequestGuid, Request> all_requests;
  std::unordered_map<RequestGuid, GenerationResult> request_generation_results;
  std::mutex request_queue_mutex;
  RequestGuid next_available_guid;
  // Legion futures for inc_decoding and spec_infer
  BatchConfigFuture last_bcf;
  InferenceResultFuture last_irf;
  TreeVerifyBatchConfigFuture last_tree_bcf;
  InferenceResultFuture last_tree_irf;

  // TODO: Move this two vector to request struct
  std::unordered_map<RequestGuid,
                     std::vector<std::pair<BatchConfig::TokenId, int>>>
      dfs_tree_inputs;
  std::unordered_map<RequestGuid, std::vector<std::pair<int, int>>>
      committed_tokens;

  // Multi-model support
  std::vector<FFModel *> models;

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

}; // namespace FlexFlow
