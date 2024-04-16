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

  // TokenTree speculative_token_tree;
  std::vector<TokenTree> speculative_token_trees;
  // To make request manager stateful, we need to store the causal mask here
  BatchConfig::BitMask causal_mask;
};

class TokenTreeNode {
public:
  BatchConfig::TokenId id;
  float joint_prob;
  int parent_pos;
  bool pruned = false;

  TokenTreeNode(BatchConfig::TokenId id, float joint_prob, int parent_pos)
      : id(id), joint_prob(joint_prob), parent_pos(parent_pos) {}
  bool operator>(TokenTreeNode const &other) const {
    return joint_prob > other.joint_prob;
  }
};

// A comparator for shared_ptr<TokenTreeNode>
struct CompareSharedTokenTreeNodePtrRequestGuidPair {
  bool operator()(std::pair<std::shared_ptr<TokenTreeNode>,
                            BatchConfig::RequestGuid> const &lhs,
                  std::pair<std::shared_ptr<TokenTreeNode>,
                            BatchConfig::RequestGuid> const &rhs) const {
    return lhs.first->joint_prob > rhs.first->joint_prob;
  }
};

class TokenTree {
public:
  std::vector<std::list<shared_ptr<TokenTreeNode>>> tree_layers = {};
  // The numebr of tokens in the tree that are not pruned
  int tree_size = 0;
  // The numebr of tokens in the tree including the pruned ones
  int tree_node_size = 0;
  void add_layer() {
    tree_layers.emplace_back();
  }
};

class RequestManager {
public:
  enum Status {
    PREFILLING = 1001,
    DECODING = 1002,
    SSM_SPEC = 1003,
    LLM_VERIFY = 1004,
  };
  enum BackgroundServerStatus {
    INITIALIZED = 2001,
    SERVING = 2002,
    TERMINATED = 2003,
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
  void appendPendingRequest(BatchConfig::BitMask &bitmask, int initLength);
  void init_bitmask(BatchConfig::BitMask &bitmask, int initLength);
  void append_bitmask(BatchConfig::BitMask &bitmask,
                      int newNodes,
                      BeamTree const tree,
                      int currentDepth);
  void append_bitmask(RequestGuid guid, BatchConfig::BitMask &bitmask);
  void update_bitmask(BatchConfig::BitMask &bitmask,
                      int initLength,
                      int non_tree_size);

  FFModel *get_ssm_model(int model_id);

  void serve_incr_decoding(FFModel *model);
  void serve_spec_infer(FFModel *model);
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
                              SsmInferenceResult const &result);
  TreeSearchBatchConfigFuture
      prepare_next_batch_beam(TreeSearchBatchConfigFuture const &old_bc,
                              SsmInferenceResultFuture const &result,
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

  TreeVerifyBatchConfig prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfig> const &old_batches);
  TreeVerifyBatchConfigFuture prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfigFuture> const &old_batches,
      Legion::Context ctx,
      Legion::Runtime *runtime);

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
  // Given the last speculation result, prepare the next speculation batch.
  TreeSearchBatchConfig
      prepare_next_batch_spec(SsmInferenceResult const &ssm_inference_result);
  // A wrapper function.
  TreeSearchBatchConfigFuture
      prepare_next_batch_spec(TreeSearchBatchConfigFuture const &old_bc,
                              SsmInferenceResultFuture const &result,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  // Given the verification result, prepare the first speculation batch.
  TreeSearchBatchConfig
      prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                              InferenceResult const &result,
                              int model_id);
  // A wrapper function.
  TreeSearchBatchConfigFuture
      prepare_next_batch_init(TreeVerifyBatchConfigFuture const &old_bc,
                              InferenceResultFuture const &result,
                              int model_id,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
  // Given the speculation result, prepare the verification batch.
  TreeSearchBatchConfig prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfig> const &old_batches);
  // A wrapper function.
  TreeSearchBatchConfigFuture prepare_next_batch_verify(
      std::vector<TreeSearchBatchConfigFuture> const &old_batches,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  // This function takes the small model inference results and the last
  // speculation batch config and use the information to update the token tree
  // stored in RequestManager::all_requests.
  void store_spec_metadata(TreeSearchBatchConfig const &old_bc,
                           SsmInferenceResult const &result);
  // Put the last layer of the token tree stored in RequestManager::all_requests
  // into new_bc::beamRequestsInfo .
  void update_spec_metadata(TreeSearchBatchConfig &new_bc,
                            TreeSearchBatchConfig const &old_bc,
                            Token &tree,
                            int request_index);

  // This function takes the tree stored in the token trees in
  // RequestManager::all_requests, and convert them into serialized version.
  // Called by prepare_next_batch_verify().
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

  /* Old APIs for reference */
  // A wrapper function.
  static TreeSearchBatchConfig prepare_next_batch_beam_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);

  // A wrapper function.
  static TreeSearchBatchConfig prepare_next_batch_init_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  /* Old APIs for reference */

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

  /* Old APIs for reference */
  static TreeVerifyBatchConfig prepare_next_batch_verify_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  /* Old APIs for reference */

  /* New APIs */
  static TreeSearchBatchConfig prepare_next_batch_verify_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  /* New APIs */

  void update_inference_results(InferenceResult const &results);
  bool update_inference_results(SsmInferenceResult const &ssm_inference_result);
  BatchConfig get_next_batch_config();

private:
  // configuration parameters
  int max_requests_per_batch;
  int max_tokens_per_batch;
  int max_spec_tree_token_num;
  int max_sequence_length;
  Status request_manager_status;
  BackgroundServerStatus background_server_status;

  // tree width in each speculative step, if not specified 1
  [[deprecated("This field will be removed")]]
  std::vector<int> spec_infer_tree_width; // Old version, delete after refactor

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

  // Added to make the request manager stateful. During the processing of the
  // first small model inference results, the step equals to 1. That is, every
  // time a small model inference task is launched, the step is increased by 1.
  int current_speculation_step = 0;
  // Maps the index of the request in the batch config to the request guid.
  int guid_of_requests[BatchConfig::MAX_NUM_REQUESTS];
  bool request_available[BatchConfig::MAX_NUM_REQUESTS];
  int num_available_requests = 0;

  // This is a helper data structure to store help the pruning of the token
  // trees across different requests.
  std::priority_queue<
      std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid>,
      std::vector<std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid>>,
      CompareSharedTokenTreeNodePtrRequestGuidPair>
      token_tree_node_pool;

  // TODO: Move this two vector to request struct
  std::unordered_map<RequestGuid,
                     std::vector<std::pair<BatchConfig::TokenId, int>>>
      dfs_tree_inputs;
  std::unordered_map<RequestGuid, std::vector<std::pair<int, int>>>
      committed_tokens;

  // Multi-model support
  std::vector<FFModel *> ssm_models;

  // Background server handler
  Legion::Future background_server_handler;

  // Performance profiling
  size_t num_processed_requests;

  struct ProfileInfo {
    int llm_decoding_steps;
    int ssm_decoding_steps;
    double start_time, finish_time;
  };
  std::unordered_map<RequestGuid, ProfileInfo> profiling_requests;
  double total_request_run_time;

  void RequestManager::initialize_root_of_spec_token_trees();
  void add_token_to_spec_token_tree(RequestGuid guid,
                                    BatchConfig::TokenId token_id,
                                    int parent_pos,
                                    float joint_prob);
  void prune_last_layer_of_spec_token_tree(RequestGuid guid);
};

}; // namespace FlexFlow
