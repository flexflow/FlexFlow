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
class TokenTree;
class RequestManager;
using tokenizers::Tokenizer;

class InferenceManager {
public:
  InferenceManager();
  static InferenceManager *get_inference_manager();
  void compile_model_and_allocate_buffer(FFModel *model, bool is_llm = true);
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

class TokenTreeNode {
public:
  BatchConfig::TokenId id;
  double log_accumulated_prob;
  int parent_pos;
  bool included = false;
  bool gumbel = false;
  float gumbel_logit = 0.0f;

  TokenTreeNode(BatchConfig::TokenId id,
                double log_accumulated_prob,
                int parent_pos,
                bool gumbel = false,
                float gumbel_logit = 0.0f)
      : id(id), log_accumulated_prob(log_accumulated_prob),
        parent_pos(parent_pos), gumbel(gumbel), gumbel_logit(gumbel_logit) {}
};

bool operator<(std::shared_ptr<TokenTreeNode> const &lhs,
               std::shared_ptr<TokenTreeNode> const &rhs);

bool operator<=(std::shared_ptr<TokenTreeNode> const &lhs,
                std::shared_ptr<TokenTreeNode> const &rhs);

// A comparator for std::shared_ptr<TokenTreeNode>
// This is used to construct a max heap for the token tree nodes
struct SharedTokenTreeNodePtrLess {
  bool operator()(std::shared_ptr<TokenTreeNode> const &lhs,
                  std::shared_ptr<TokenTreeNode> const &rhs) const {
    if (lhs->gumbel) {
      assert(rhs->gumbel);
      return lhs->gumbel_logit < rhs->gumbel_logit;
    }
    return lhs->log_accumulated_prob < rhs->log_accumulated_prob;
  }
};

class TokenTree {
public:
  std::list<std::list<shared_ptr<TokenTreeNode>>> tree_layers = {};
  void add_layer() {
    tree_layers.emplace_back();
  }

  void clear() {
    tree_layers.clear();
  }
};

std::ostream &operator<<(std::ostream &os, TokenTree const &token_tree);

struct Request {
  enum Status {
    PENDING = 101,   // loading prompt
    RUNNING = 102,   // running inference
    COMPLETED = 103, // finished and verified
    FINISHING = 104, // finishing request, but not yet verified
  };
  BatchConfig::RequestGuid guid;
  int batch_index = -1;
  int ssm_cache_size = 0;
  int llm_cache_size = 0;
  double slo_ratio = 1.0;
  double decode_latency_ms = 0.0;
  int ssm_prefill_len = 0;
  int llm_prefill_len = 0;
  bool attained = false;

  int first_token_offset_in_batch = 0;
  int num_tokens_in_batch = 0;

  Status status = PENDING;
  std::vector<BatchConfig::TokenId> tokens;

  // TokenTree speculative_token_tree;
  std::vector<TokenTree> speculative_token_trees;
  // To make request manager stateful, we need to store the causal mask here
  BatchConfig::BitMask causal_mask;
  // Here we maintain a struct CommittedToken which has a field `from_index` and
  // `to_index`. The `from_index` is used by the LLM KV cache commitment and the
  // `to_index` is used both by the the SSM KV cache recomputation and the LLM
  // KV cache commitment. Details are as follows:
  //
  // 1. Recompute the SSM KV cache: We don't commit the KV cache of the SSM
  // committed tokens but recompute them instead. That is, after the we append
  // the committed tokens to the generated sequence, just like in the prefilling
  // phase, and pass them into the SSM to recompute the KV cache. Here we don't
  // need `from_index` because we don't copy the KV cache, but we need
  // `to_index`, which is the indices of the committed tokens in the request.
  //
  // to_index -> BatchConfig::PerTokenInfo.abs_index_in_request
  //
  // 2. Commit the LLM KV cache: On the GPU, the KV cache of the speculative
  // token tree and the generated tokens are stored separately. So the
  // `from_index` should be the index of the token in the speculative token
  // tree. `to_index` should be the place to put the KV cache in the LLM KV
  // cache: prompt_length + generated_sequence_length +
  // index_in_committed_tokens.
  //
  // from_index -> TreeVerifyBatchConfig::CommittedTokensInfo.index_in_kv_cache
  // to_index -> TreeVerifyBatchConfig::CommittedTokensInfo.token_depth
  //
  // Actually, for a committed token, the `to_index` for the LLM KV cache and
  // the SSM KV cache are the same thing, so we can use the same field to store
  // the information.
  //
  // When storing the committed tokens:
  // from_index: The offset of the committed token in the request in the
  // TreeVerifyBatchConfig
  // to_index: The absolute index of the token in the request

  struct CommittedToken {
    int from_index;
    int to_index;
    BatchConfig::TokenId token_id;
    CommittedToken(int from_index, int to_index, BatchConfig::TokenId token_id)
        : from_index(from_index), to_index(to_index), token_id(token_id) {}
  };
  std::vector<CommittedToken> committed_tokens;

  // Enabling Streaming KVCache means we doesn't store the whole KV sequence of
  // the tokens in a request. Instead, we only store the sink cache (a few
  // foremost tokens) and the window cache (rolling-updated backmost tokens
  // through decoding). Currently, we only use streaming cache in the *draft
  // model* calculation.
  // - Maintain the streaming cache: During inference, we
  // first fill up the sink cache then the window cache. After the window cache
  // is full, we move back to the beginning of the window cache and commit the
  // tokens in replace there.
  // - When to update the streaming cache:
  // 1. Prefilling phase
  // 2. Committing phase after the target model verification
  StreamingCacheInfo streaming_cache_info;

  std::priority_queue<std::shared_ptr<TokenTreeNode>,
                      std::vector<std::shared_ptr<TokenTreeNode>>,
                      SharedTokenTreeNodePtrLess>
      token_tree_nodes_pq;

  double get_length_weight();
  void set_slo_ratio(double slo_ratio_);
  double get_slo_ratio();
};

class RequestManager {
public:
  enum State {
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
  enum DecodingMode {
    INCREMENTAL_DECODING = 3001,
    SPECULATIVE_DECODING = 3002,
  };
  enum PrefillModel {
    LLM = 4001,
    SSM = 4002,
  };

  using RequestGuid = BatchConfig::RequestGuid;
  using TokenId = BatchConfig::TokenId;

  inline static RequestGuid const INVALID_GUID = 0;
  RequestManager();
  static RequestManager *get_request_manager();
  size_t get_num_processed_requests();
  size_t get_num_ssms();

  void set_max_requests_per_batch(int max_num_requests);
  int get_max_requests_per_batch();
  void set_max_tokens_per_batch(int max_num_tokens);
  int get_max_tokens_per_batch();
  void set_max_tokens_per_ssm_batch(int max_num_ssm_tokens);
  int get_max_tokens_per_ssm_batch();
  int get_max_spec_tree_token_num();
  void set_max_sequence_length(int max_seq_length);
  int get_max_sequence_length();
  void set_decoding_mode(DecodingMode mode);
  void set_verbose(bool verbose_);
  int get_k();
  void set_k(int k);
  int get_max_tree_depth();
  void set_max_tree_depth(int max_tree_depth);
  int get_max_tree_width();
  void set_max_tree_width(int max_tree_width);
  void set_speculative_sampling(bool speculative_sampling);
  void set_baseline_latency(double baseline_latency_ms);
  double get_baseline_latency();
  void set_ssm_spec_latency(double ssm_spec_latency_ms);
  double get_ssm_spec_latency();
  void set_llm_verify_latency(double llm_verify_latency_ms);
  double get_llm_verify_latency();
  void set_correction_factor(double correction_factor);
  double get_correction_factor();
  void set_streaming_cache(bool streaming_cache);
  bool get_memory_occupancy();
  void set_memory_occupancy(bool memory_occupancy);
  void
      set_slo_violation_early_termination(bool slo_violation_early_termination);
  double get_request_expected_latency(Request &request);
  Request &get_request_with_guid(RequestGuid guid);
  int register_ssm_model(FFModel *model);
  void register_tokenizer(ModelType model_type,
                          int bos_token_id,
                          int eos_token_id,
                          std::string const &path);
  void register_output_filepath(std::string const &);

  FFModel *get_ssm_model(int model_id);

  void serve_spec_infer(FFModel *model);
  void serve_spec_infer_sync(FFModel *model);
  void serve_decoding(FFModel *model);
  GenerationResult get_generation_result(RequestGuid const &guid);
  RequestGuid register_new_request(GenerationRequest const &req);
  // Methods to start and terminate request manager's background task
  void start_background_server(FFModel *model);
  bool is_background_server_serving();
  bool is_background_server_terminated();
  void terminate_background_server();
  static void terminate_background_server_at_exit();
  // Methods to check and mark request completion
  bool is_request_completed(RequestGuid const &guid);
  void trigger_request_completion_future(RequestGuid const &guid);
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
  // API for rm state machine
  BatchConfigFuture get_next_batch_config(InferenceResultFuture const &result,
                                          Legion::Context ctx,
                                          Legion::Runtime *runtime);
  static BatchConfig get_next_batch_config_task(
      Legion::Task const *task,
      std::vector<Legion::PhysicalRegion> const &regions,
      Legion::Context ctx,
      Legion::Runtime *runtime);
  BatchConfig get_next_batch_config(InferenceResult const &result);
  void update_inference_results(InferenceResult const &result);
  BatchConfig prepare_next_batch();

  int get_num_active_requests();
  int get_empty_request_index();

  // Comparters
  struct SharedTokenTreeNodePtrRequestGuidWeightedLess {
    bool operator()(
        std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid> const &lhs,
        std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid> const &rhs)
        const;
  };
  struct SharedTokenTreeNodePtrRequestGuidLess {
    bool operator()(
        std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid> const &lhs,
        std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid> const &rhs)
        const;
  };

private:
  // configuration parameters
  int max_requests_per_batch;
  int max_tokens_per_batch;
  int max_tokens_per_ssm_batch;
  int max_spec_tree_token_num;
  int max_sequence_length;
  int max_tree_depth;
  int max_tree_width;
  int k;
  // Profile based latency
  double baseline_latency_ms = 43;
  double ssm_spec_latency_ms = 17;
  double llm_verify_latency_ms = 65;
  double correction_factor = 1.05;

  State request_manager_status;
  BackgroundServerStatus background_server_status;
  DecodingMode decoding_mode;
  PrefillModel prefill_model;
  bool speculative_sampling = false;
  // specify if enable streaming cache for incremental decoding or draft model
  bool streaming_cache = false;
  bool memory_occupancy = false;
  bool slo_violation_early_termination = false;

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
  Request *prefill_request = nullptr;

  // Added to make the request manager stateful. During the processing of the
  // first small model inference results, the step equals to 1. That is, every
  // time a small model inference task is launched, the step is increased
  // by 1.
  int current_ssm_step = 0;
  // Maps the index of the request in the batch config to the request guid.
  int guid_of_requests[BatchConfig::MAX_NUM_REQUESTS];
  bool request_available[BatchConfig::MAX_NUM_REQUESTS];
  int num_available_requests = 0;
  int ssm_completed = true;

  // rm state
  std::mutex rm_state_mutex;

  // Multi-model support
  std::vector<FFModel *> ssm_models;

  // Background server handler
  Legion::Future background_server_handler;

  // Performance profiling
  // TODO: maintain this field
  size_t num_processed_requests;

  struct RequestProfileInfo {
    int llm_prefilling_steps = 0;
    int ssm_prefilling_steps = 0;
    int llm_decoding_steps = 0;
    int ssm_decoding_steps = 0;
    long long start_time = 0, start_decoding_time = 0, finish_time = 0;
  };
  struct ProfileInfo {
    // For SpecInfer: One step is comprised of one ssm speculation phase + a
    // single llm verification phase (forward pass + verification) For Incr
    // Decoding: One step is one LLM decoding phase
    long long llm_step_start = 0, ssm_step_start = 0;
    // Times for each LLM verification phase (in ms)
    std::vector<double> llm_step_times;
    // Number of requests in batch at each step
    std::vector<int> requests_per_step;
    // Times for each SSM speculation phase (in ms)
    std::vector<double> ssm_step_times;
    // Number of requests getting decoded at each step
    std::vector<int> ssm_steps;
    // Number of generated tokens at each step
    std::vector<int> generated_tokens_per_step;
    // To calculate the E2E time of serving
    long long server_start_time = 0;
  };

  ProfileInfo profiling;
  std::unordered_map<RequestGuid, RequestProfileInfo> profiling_requests;
  double total_request_run_time;
  bool load_pending_request_to_batch();
  void request_complete_clean_up(int batch_index, bool attained);
  /* ---------- Incremental Decoding Helper Functions ---------- */
  bool update_llm_prefill_results(InferenceResult const &result);
  bool update_llm_decode_results(InferenceResult const &result);
  BatchConfig prepare_llm_prefilling_batch();
  BatchConfig prepare_decoding_batch();
  /* ---------- Incremental Decoding Helper Functions ---------- */

  /* ---------- Spec Decoding Helper Functions ---------- */
  BatchConfig prepare_ssm_prefilling_batch();
  bool update_llm_verify_results(InferenceResult const &llm_verify_result);
  bool
      update_ssm_inference_results(InferenceResult const &ssm_inference_result);
  void update_ssm_prefill_results(InferenceResult const &ssm_prefill_result);
  // Prepare the next speculation batch config. This function is called before
  // the second step of the speculation.
  BatchConfig prepare_next_spec_batch_config();
  // Prepare the first speculation batch config. This function is called before
  // the first step of the speculation. The difference with
  // prepare_next_batch_config_spec is that we put the info of the committed
  // tokens into the batch config in the first speculation step to commit the KV
  // cache of the small model.
  BatchConfig prepare_first_spec_batch_config();
  BatchConfig prepare_verify_batch_config();

  // LLM result verification
  void get_verify_results_greedy(InferenceResult const &llm_verify_result);
  void get_verify_results_sample(InferenceResult const &llm_verify_result);

  // Bitmask related
  void init_bitmask_prompt(RequestGuid guid, int prompt_length);
  void append_bitmask(RequestGuid guid);
  void update_bitmask_prompt(RequestGuid guid, int num_committed_tokens);
  void init_bitmask_spec(RequestGuid guid);
  BatchConfig::BitMask create_llm_bitmask(RequestGuid guid);

  // Token tree related
  void init_token_tree(RequestGuid guid);
  void add_root_to_spec_token_tree(RequestGuid guid,
                                   BatchConfig::TokenId token_id);
  void add_tokens_to_spec_token_tree(
      InferenceResult const &ssm_inference_result);
  void prune_token_tree();
  void add_tokens_toward_slo(RequestGuid guid, int &budget);
  void add_tokens_toward_memory_occupancy(int budget);
  void add_tokens_toward_goodput(int budget);

  /* ---------- Spec Decoding Helper Functions ---------- */
  void renormalize(std::vector<std::pair<TokenId, float>> &D,
                   std::unordered_map<TokenId, float> &R,
                   TokenId token_id);
  std::tuple<int, BatchConfig::TokenId, bool>
      reject_sampling(std::vector<std::pair<TokenId, float>> &D,
                      std::unordered_map<TokenId, float> &R,
                      int k);
  void gumbel_conditioned_on_max(double target_max,
                                 std::vector<std::pair<double, int>> &logits);

  // Profiling related functions
  void reset_profiling_statistics();
};
}; // namespace FlexFlow
