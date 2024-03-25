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

namespace FlexFlow {

class FFModel;
class BeamTree;
class RequestManager;
using tokenizers::Tokenizer;

class InferenceManager {
public:
  InferenceManager();
  static InferenceManager *get_inference_manager();
  void compile_model_and_allocate_buffer(FFModel *model);
  void init_operators_inference(FFModel *model);
  Legion::FutureMap inference(FFModel *model, int index, BatchConfig const &bc);
  Legion::FutureMap
      inference(FFModel *model, int index, BatchConfigFuture const &bc);
  void load_input_tokens_from_batch_config(FFModel *model,
                                           BatchConfigFuture const &bc,
                                           ParallelTensor const input,
                                           FFHandler *handlers);
  void load_positions(FFModel *model,
                      BatchConfigFuture const &bc,
                      ParallelTensor position_input,
                      int offset);
  void register_model_weights_loader(FFModel *, FileDataLoader *);
  void load_inference_metadata_batch_config(FFModel *model,
                                            BatchConfigFuture const &bc,
                                            FFHandler *handlers);

public:
  std::unordered_map<ParallelTensor, std::vector<ParallelTensor>> tensor_buffer;
  std::unordered_map<FFModel *, FileDataLoader *> model_weights_loaders;
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

  std::vector<struct BeamTree> beam_trees; // Old version, delete after refactor
  std::vector<struct TokenTree> token_trees; // New version
};

// The old version of beam tree
// store the result of beam search
struct BeamTree {
  struct treeLayer {
    TreeSearchBatchConfig::TokenId
        tokens[TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    int parent_ids[TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    float probs[TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES];
    int nodes_num_this_layer = 0;
  };
  treeLayer treeLayers[TreeSearchBatchConfig::MAX_BEAM_DEPTH + 1];
};

// The new version of BeamTree
// Named as TokenTree, supports general tree structure.
class TokenTree {
  class Node {
  public:
    BatchConfig::TokenId id;
    float unconditional_prob;
    std::vector<std::shared_ptr<Node>> children;
    std::shared_ptr<Node> parent;
    Node(BatchConfig::TokenId id, float prob, std::shared_ptr<Node> parent)
        : id(id), unconditional_prob(prob), parent(parent) {}
  };

  class TreeLayer {
  public:
    std::vector<std::shared_ptr<Node>> nodes;
  };

private:
  std::vector<TreeLayer> layers;
  // Do we need the root?
  std::shared_ptr<Node> root;

public:
  TokenTree(BatchConfig::TokenId root_id, float root_prob)
      : root(std::make_shared<Node>(root_id, root_prob, nullptr)) {
    layers.push_back(TreeLayer());
    layers[0].nodes.push_back(root);
  }
};

class RequestManager {
public:
  enum Status {
    INITIALIZED = 1001,
    SERVING = 1002,
    TERMINATED = 1003,
  };
  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;

  static RequestGuid const INVALID_GUID = 0;
  RequestManager();
  static RequestManager *get_request_manager();
  size_t get_num_processed_requests();
  size_t get_num_ssms();

  void set_max_requests_per_batch(int max_num_requests);
  int get_max_requests_per_batch();
  void set_max_tokens_per_batch(int max_num_tokens);
  int get_max_tokens_per_batch();
  void set_max_spec_tree_token_num(int max_num_tokens);
  int get_max_spec_tree_token_num();
  int get_max_verify_tokens_per_batch();
  void set_max_sequence_length(int max_seq_length);
  void push_spec_infer_tree_width(int tree_width);
  int get_max_sequence_length();
  int register_ssm_model(FFModel *model);
  void register_tokenizer(ModelType model_type,
                          int bos_token_id,
                          int eos_token_id,
                          std::string const &path);
  void register_output_filepath(std::string const &);
  void initBitMask(BatchConfig::BitMask &bitmask, int initLength);
  void appendPendingRequest(BatchConfig::BitMask &bitmask, int initLength);
  void appendBitMask(BatchConfig::BitMask &bitmask,
                     int newNodes,
                     int preBeamSize,
                     int old_sub_num,
                     BeamTree const tree,
                     int currentDepth);
  void updateBitMask(BatchConfig::BitMask &bitmask,
                     int initLength,
                     int non_tree_size);

  FFModel *get_ssm_model(int model_id);

  void serve_incr_decoding(FFModel *model);
  void serve_spec_infer(FFModel *model);
  void serve_spec_infer_v2(FFModel *model);
  GenerationResult get_generation_result(RequestGuid const &guid);
  RequestGuid register_new_request(std::string const &prompt,
                                   int max_sequence_length);
  RequestGuid register_new_request(std::vector<TokenId> const &prompt,
                                   int max_sequence_length);
  // Methods to start and terminate request manager's background task
  void start_background_server(FFModel *model);
  bool is_background_server_terminated();
  void terminate_background_server();
  static void terminate_background_server_at_exit();
  // Methods to check and mark request completion
  bool is_request_completed(RequestGuid const &guid);
  void trigger_request_completion_future(RequestGuid const &guid);

  // Methods for preparing next batches
  BatchConfig prepare_next_batch(BatchConfig const &bc,
                                 InferenceResult const &result);
  BatchConfigFuture prepare_next_batch(BatchConfigFuture const &bc,
                                       InferenceResultFuture const &result,
                                       Legion::Context ctx,
                                       Legion::Runtime *runtime);
  /* Old APIs for reference */
  TreeSearchBatchConfig
      prepare_next_batch_beam(TreeSearchBatchConfig const &old_bc,
                              BeamInferenceResult const &result);
  BeamSearchBatchConfigFuture
      prepare_next_batch_beam(BeamSearchBatchConfigFuture const &old_bc,
                              BeamInferenceResultFuture const &result,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  TreeSearchBatchConfig
      prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                              InferenceResult const &result,
                              int model_id);
  BeamSearchBatchConfigFuture
      prepare_next_batch_init(TreeVerifyBatchConfigFuture const &old_bc,
                              InferenceResultFuture const &result,
                              int model_id,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);

  TreeVerifyBatchConfig prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfig> const &old_batches);
  TreeVerifyBatchConfigFuture prepare_next_batch_verify(
      std::vector<BeamSearchBatchConfigFuture> const &old_batches,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  void store_beam_metadata(TreeSearchBatchConfig const &old_bc,
                           BeamInferenceResult const &result);
  void update_beam_metadata(TreeSearchBatchConfig &new_bc,
                            TreeSearchBatchConfig const &old_bc,
                            BeamTree &tree,
                            int request_index);

  std::vector<std::pair<BatchConfig::TokenId, int>>
      traverse_beam_tree(TreeSearchBatchConfig const &old_bc,
                         int request_index,
                         int first_token_depth_in_request);
  /* Old APIs for reference */

  /* New APIs */
  TreeSearchBatchConfig
      prepare_next_batch_spec(TreeSearchBatchConfig const &old_bc,
                              BeamInferenceResult const &result);
  TreeSearchBatchConfigFuture
      prepare_next_batch_spec(TreeSearchBatchConfigFuture const &old_bc,
                              BeamInferenceResultFuture const &result,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  TreeSearchBatchConfig
      prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                              InferenceResult const &result,
                              int model_id);
  TreeSearchBatchConfigFuture
      prepare_next_batch_init(TreeVerifyBatchConfigFuture const &old_bc,
                              InferenceResultFuture const &result,
                              int model_id,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  TreeSearchBatchConfig prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfig> const &old_batches);
  TreeVerifyBatchConfigFuture prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfigFuture> const &old_batches,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  void store_spec_metadata(TreeSearchBatchConfig const &old_bc,
                           BeamInferenceResult const &result);
  void update_spec_metadata(TreeSearchBatchConfig &new_bc,
                            TreeSearchBatchConfig const &old_bc,
                            Token &tree,
                            int request_index);

  std::vector<std::pair<BatchConfig::TokenId, int>>
      traverse_spec_tree(TreeSearchBatchConfig const &old_bc,
                         int request_index,
                         int first_token_depth_in_request);
  /* New APIs */

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
  static void background_serving_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
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

  static void
      load_batch_config_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static BatchConfig prepare_next_batch_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  /* APIs to modify */
  static TreeSearchBatchConfig prepare_next_batch_beam_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static TreeSearchBatchConfig prepare_next_batch_init_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  /* APIs to modify */

  /* New APIs */
  static TreeSearchBatchConfig prepare_next_batch_spec_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  static TreeSearchBatchConfig prepare_next_batch_init_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  /* New APIs */

  static TreeVerifyBatchConfig prepare_next_batch_verify_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

private:
  // configuration parameters
  int max_requests_per_batch;
  int max_tokens_per_batch;
  int max_spec_tree_token_num;
  int max_sequence_length;
  Status request_manager_status;

  // tree width in each speculative step, if not specified 1
  std::vector<int> spec_infer_tree_width; // Old version, delete after refactor

  // private fields
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
  std::unordered_map<RequestGuid, std::promise<void> *> request_to_promise;
  std::mutex request_to_promise_mutex;
  RequestGuid next_available_guid;

  // TODO: Move this two vector to request struct
  std::unordered_map<RequestGuid,
                     std::vector<std::pair<BatchConfig::TokenId, int>>>
      dfs_tree_inputs;
  std::unordered_map<RequestGuid, std::vector<std::pair<int, int>>>
      committed_tokens;

  // Multi-model support
  std::vector<FFModel *> ssm_models;

  // Performance profiling
  size_t num_processed_requests;

  // Background server handler
  Legion::Future background_server_handler;

private:
  struct ProfileInfo {
    int llm_decoding_steps;
    int ssm_decoding_steps;
    double start_time, finish_time;
  };
  std::unordered_map<RequestGuid, ProfileInfo> profiling_requests;
  double total_request_run_time;
};

}; // namespace FlexFlow
