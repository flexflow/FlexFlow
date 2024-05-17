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

#include "flexflow/request_manager.h"
#include "flexflow/parallel_ops/parallel_op.h"
// #include "flexflow/tokenizers.h"
#include <bitset>
#include <cmath>
#include <filesystem>
#include <future>
#include <iomanip>
#include <new>
#include <stack>
#include <stdexcept>

namespace FlexFlow {

using namespace Legion;
using tokenizers::Tokenizer;

LegionRuntime::Logger::Category log_req_mgr("RequestManager");

std::string LoadBytesFromFile(std::string const &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  assert(fs.is_open() && "Failed to open file for reading.");
  fs.seekg(0, std::ios::end);
  size_t size = fs.tellg();
  fs.seekg(0, std::ios::beg);
  std::string data(size, '\0');
  fs.read(&data[0], size);
  assert(!fs.fail() && "Failed to read data from file.");
  return data;
}

RequestManager::RequestManager()
    : background_server_status(INITIALIZED), verbose(false),
      next_available_guid(1000000), num_processed_requests(0),
      total_request_run_time(0.0f), request_manager_status(PREFILLING),
      decoding_mode(INCREMENTAL_DECODING), prefill_model(SSM) {
  // The following config parameters are set
  // during ffmodel.compile()
  // Initialize them to -1 to make sure no one
  // gets an incorrect value of them before
  // ffmodel.compile()
  max_requests_per_batch = -1;
  max_tokens_per_batch = -1;
  max_spec_tree_token_num = -1;
  max_sequence_length = -1;
  max_tree_depth = -1;
  max_tree_width = -1;
  k = -1;
  std::fill(std::begin(request_available), std::end(request_available), false);
  std::fill(
      std::begin(guid_of_requests), std::end(guid_of_requests), INVALID_GUID);
}

void RequestManager::set_max_requests_per_batch(int max_num_requests) {
  assert(max_requests_per_batch == -1 ||
         max_requests_per_batch == max_num_requests);
  max_requests_per_batch = max_num_requests;
  assert(max_requests_per_batch <= BatchConfig::MAX_NUM_REQUESTS);
}

int RequestManager::get_max_requests_per_batch() {
  assert(max_requests_per_batch > 0);
  return max_requests_per_batch;
}

void RequestManager::set_max_tokens_per_batch(int max_num_tokens) {
  assert(max_tokens_per_batch == -1 || max_tokens_per_batch == max_num_tokens);
  max_tokens_per_batch = max_num_tokens;
  assert(max_tokens_per_batch <= BatchConfig::MAX_NUM_TOKENS);
}

void RequestManager::set_max_spec_tree_token_num(int max_num_tokens) {
  assert(max_spec_tree_token_num == -1 ||
         max_spec_tree_token_num == max_num_tokens);
  max_spec_tree_token_num = max_num_tokens;
  assert(max_spec_tree_token_num <= BatchConfig::MAX_SPEC_TREE_TOKEN_NUM);
}

int RequestManager::get_max_tokens_per_batch() {
  assert(max_tokens_per_batch > 0);
  return max_tokens_per_batch;
}

int RequestManager::get_max_spec_tree_token_num() {
  assert(max_spec_tree_token_num > 0);
  return max_spec_tree_token_num;
}

int RequestManager::get_max_verify_tokens_per_batch() {
  assert(max_tokens_per_batch > 0);
  return max_tokens_per_batch;
}

void RequestManager::set_max_sequence_length(int max_seq_length) {
  assert(max_sequence_length == -1 || max_sequence_length == max_seq_length);
  max_sequence_length = max_seq_length;
}

int RequestManager::get_max_sequence_length() {
  assert(max_sequence_length > 0);
  return max_sequence_length;
}

void RequestManager::set_decoding_mode(DecodingMode mode) {
  assert(mode == INCREMENTAL_DECODING || mode == SPECULATIVE_DECODING);
  decoding_mode = mode;
}

void RequestManager::set_verbose(bool verbose_) {
  verbose = verbose_;
}

int RequestManager::get_k() {
  assert(k > 0 and k <= BatchConfig::MAX_SPEC_TREE_TOKEN_NUM and "Invalid k");
  return k;
}

void RequestManager::set_k(int _k) {
  assert(_k > 0 and _k <= BatchConfig::MAX_SPEC_TREE_TOKEN_NUM and "Invalid k");
  k = _k;
}

int RequestManager::get_max_tree_depth() {
  assert(max_tree_depth > 0 and
         max_tree_depth <= BatchConfig::MAX_TREE_DEPTH and
         "Invalid max_tree_depth");
  return max_tree_depth;
}

void RequestManager::set_max_tree_depth(int max_tree_depth) {
  assert(max_tree_depth > 0 and
         max_tree_depth <= BatchConfig::MAX_TREE_DEPTH and
         "Invalid max_tree_depth");
  this->max_tree_depth = max_tree_depth;
}

int RequestManager::get_max_tree_width() {
  assert(max_tree_width > 0 and
         max_tree_width <= BatchConfig::MAX_TREE_WIDTH and
         "Invalid max_tree_width");
  return max_tree_width;
}

void RequestManager::set_max_tree_width(int max_tree_width) {
  assert(max_tree_width > 0 and
         max_tree_width <= BatchConfig::MAX_TREE_WIDTH and
         "Invalid max_tree_width");
  this->max_tree_width = max_tree_width;
}

void RequestManager::register_tokenizer(ModelType type,
                                        int bos_token_id,
                                        int eos_token_id,
                                        std::string const &path) {
  this->model_type = type;
  this->bos_token_id = bos_token_id;
  this->eos_token_id = eos_token_id;
  std::string tokenizer_folder =
      (!path.empty() && path.back() != '/') ? path + '/' : path;
  if (model_type == ModelType::LLAMA) {
    bool path_to_file = !path.empty() &&
                        (path.size() >= strlen("tokenizer.model")) &&
                        path.find("tokenizer.model") ==
                            (path.size() - strlen("tokenizer.model"));
    std::string tokenizer_filepath =
        path_to_file ? path : tokenizer_folder + "tokenizer.model";
    this->tokenizer_ =
        Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(tokenizer_filepath));
  } else if (model_type == ModelType::OPT) {
    std::string vocab_file = tokenizer_folder + "vocab.json";
    std::string merges_file = tokenizer_folder + "merges.txt";
    std::string added_tokens_file =
        tokenizer_folder + "special_tokens_map.json";
    std::filesystem::path path1(vocab_file);
    std::filesystem::path path2(merges_file);
    std::filesystem::path path3(added_tokens_file);
    assert(std::filesystem::exists(path1) &&
           "Vocab file vocab.json does not exist at the specified path");
    assert(std::filesystem::exists(path2) &&
           "Merge file merges.txt does not exist at the specified path");
    // opt_tokenizer = new OptTokenizer(vocab_file, merges_file);
    std::string vocab = LoadBytesFromFile(path1.string());
    std::string merges = LoadBytesFromFile(path2.string());
    std::string added_tokens = LoadBytesFromFile(path3.string());

    this->tokenizer_ =
        Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens);
  } else if (model_type == ModelType::FALCON ||
             model_type == ModelType::STARCODER ||
             model_type == ModelType::MPT) {
    std::string falcon_tokenizer_path = join_path({path, "tokenizer.json"});
    this->tokenizer_ =
        Tokenizer::FromBlobJSON(LoadBytesFromFile(falcon_tokenizer_path));
  }
}

void RequestManager::register_output_filepath(
    std::string const &_output_filepath) {
  this->output_filepath = _output_filepath;
}

int RequestManager::register_ssm_model(FFModel *model) {
  int model_id = ssm_models.size();
  ssm_models.push_back(model);
  std::cout << "Register new ssm model with id: " << model_id << std::endl;
  return model_id;
}

FFModel *RequestManager::get_ssm_model(int model_id) {
  assert(model_id >= 0 && model_id < ssm_models.size());
  return ssm_models[model_id];
}

size_t RequestManager::get_num_ssms() {
  return ssm_models.size();
}

RequestManager::RequestGuid
    RequestManager::register_new_request(std::vector<TokenId> const &prompt) {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);

  // Add a new request
  Request request;
  request.status = Request::PENDING;
  request.guid = next_available_guid++;

  if (prompt.size() >= get_max_sequence_length()) {
    std::cout << "Warning: too many tokens in prompt, only load up to "
              << get_max_sequence_length() << " tokens, but got "
              << prompt.size() << ".\n";

    printf("tokens size: %zu\n", request.tokens.size());
    return INVALID_GUID;
  } else {
    request.tokens = prompt;
  }

  if (get_num_ssms() == 0) {
    std::cout << "No small speculative model registered, using incremental "
                 "decoding."
              << std::endl;
  } else {
    std::cout << "Num of SSMs: " << get_num_ssms() << std::endl;
    assert(get_num_ssms() == 1 && "Only one SSM is supported now.");
    init_token_tree(request.guid);
  }

  pending_request_queue.push(request);
  all_requests[request.guid] = request;
  {
    std::lock_guard<std::mutex> const lock(request_to_promise_mutex);
    request_to_promise[request.guid] = new std::promise<void>();
  }

  if (verbose) {
    std::cout << "new req: " << request.tokens.size() << std::endl;
    for (int i = 0; i < request.tokens.size(); i++) {
      std::cout << i << " : " << request.tokens[i] << std::endl;
    }
  }

  GenerationResult gr;
  gr.guid = request.guid;
  gr.input_text = "";
  gr.input_tokens = prompt;
  gr.output_text = "";
  gr.output_tokens = prompt;
  request_generation_results[request.guid] = gr;

  return request.guid;
}

RequestManager::RequestGuid
    RequestManager::register_new_request(std::string const &prompt) {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);
  // Add a new request
  Request request;
  request.status = Request::PENDING;
  request.guid = next_available_guid++;
  if (bos_token_id >= 0 && model_type != ModelType::FALCON) {
    request.tokens.push_back(bos_token_id);
  }
  std::vector<int32_t> tokens = this->tokenizer_->Encode(prompt);
  if (tokens.size() >= get_max_sequence_length()) {
    std::cout << "Warning: too many tokens in prompt, only load up to "
              << get_max_sequence_length() << " tokens, but got "
              << tokens.size() << ".\n";

    printf("tokens size: %zu\n", tokens.size());
    return INVALID_GUID;
  }
  for (int i = 0; i < tokens.size(); i++) {
    std::cout << "[" << i << "]" << tokens.at(i) << "\n";
  }
  request.tokens.insert(request.tokens.end(), tokens.begin(), tokens.end());

  if (get_num_ssms() == 0) {
    std::cout << "No small speculative model registered, using incremental "
                 "decoding."
              << std::endl;
  } else {
    std::cout << "Num of SSMs: " << get_num_ssms() << std::endl;
    assert(get_num_ssms() == 1 && "Only one SSM is supported now.");
    init_token_tree(request.guid);
  }

  pending_request_queue.push(request);
  all_requests[request.guid] = request;
  {
    std::lock_guard<std::mutex> const lock(request_to_promise_mutex);
    request_to_promise[request.guid] = new std::promise<void>();
  }

  {
    std::string output = "New request tokens:";
    output = "[" + std::to_string(request.guid) + "]" + output;
    for (int i = 0; i < request.tokens.size(); i++) {
      output = output + " " + std::to_string(request.tokens[i]);
    }
    log_req_mgr.print("%s", output.c_str());
  }

  GenerationResult gr;
  gr.guid = request.guid;
  gr.input_text = prompt;
  gr.input_tokens = request.tokens;
  gr.output_text = prompt;
  gr.output_tokens = request.tokens;
  request_generation_results[request.guid] = gr;
  return request.guid;
}

bool RequestManager::is_request_completed(RequestGuid const &guid) {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);
  assert(all_requests.find(guid) != all_requests.end());
  Request const &request = all_requests[guid];
  // return request.tokens.size() >= request.max_sequence_length;
  return request.status == Request::COMPLETED;
}

GenerationResult
    RequestManager::get_generation_result(RequestGuid const &guid) {
  // First get the future of the request
  std::future<void> future;
  {
    std::lock_guard<std::mutex> const lock(request_to_promise_mutex);
    assert(request_to_promise.find(guid) != request_to_promise.end());
    future = request_to_promise[guid]->get_future();
  }
  // Wait until the result is completed
  future.get();
  // Get the generation result
  {
    std::lock_guard<std::mutex> const lock(request_queue_mutex);
    assert(request_generation_results.find(guid) !=
           request_generation_results.end());
    return request_generation_results[guid];
  }
}

size_t RequestManager::get_num_processed_requests() {
  return num_processed_requests;
}

int RequestManager::get_num_active_requests() {
  int count = 0;
  for (int i = 0; i < get_max_requests_per_batch(); i++) {
    if (guid_of_requests[i] != INVALID_GUID) {
      count++;
    }
  }
  return count;
}

int RequestManager::get_empty_request_index() {
  for (int i = 0; i < get_max_requests_per_batch(); i++) {
    if (guid_of_requests[i] == INVALID_GUID) {
      return i;
    }
  }
  return -1;
}

BatchConfigFuture RequestManager::get_next_batch_config(
    InferenceResultFuture const &result, Context ctx, Runtime *runtime) {
  RequestManager *rm = this;
  TaskLauncher launcher(RM_GET_NEXT_BATCH_CONFIG_TASK_ID,
                        TaskArgument(&rm, sizeof(RequestManager *)));
  launcher.add_future(result);
  return runtime->execute_task(ctx, launcher);
}

BatchConfig RequestManager::get_next_batch_config_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  RequestManager *rm = *((RequestManager **)task->args);
  InferenceResult const &result =
      Future(task->futures[0]).get_result<InferenceResult>();
  return rm->get_next_batch_config(result);
}

BatchConfig
    RequestManager::get_next_batch_config(InferenceResult const &result) {
  update_inference_results(result);
  return prepare_next_batch();
}

void RequestManager::load_pending_reqeust_to_batch() {
  assert(!pending_request_queue.empty() && "No pending request to process.");
  RequestGuid guid = pending_request_queue.front().guid;
  pending_request_queue.pop();

  prefill_request = &all_requests[guid];
  prefill_request->status = Request::RUNNING;

  // Find an empty slot
  int request_index = get_empty_request_index();
  assert(request_index != -1 && "No empty request slot to load the request.");
  // Load request into batch
  prefill_request->batch_index = request_index;
  guid_of_requests[request_index] = guid;
  request_available[request_index] = true;
  num_available_requests++;
  // Initialize the bitmask for the new request with its prompt length
  init_bitmask_prompt(guid, prefill_request->tokens.size());

  profiling_requests[guid] = ProfileInfo();
  profiling_requests[guid].start_time =
      Realm::Clock::current_time_in_microseconds();
}

void RequestManager::request_complete_clean_up(int batch_index) {
  RequestGuid guid = guid_of_requests[batch_index];
  profiling_requests[guid].finish_time =
      Realm::Clock::current_time_in_microseconds();
  Request &request = all_requests[guid];
  guid_of_requests[batch_index] = INVALID_GUID;
  request_available[batch_index] = false;
  num_available_requests--;
  request.status = Request::COMPLETED;

  std::string output = this->tokenizer_->Decode(request.tokens);
  std::cout << "Request " << guid << " completed: " << std::endl
            << output << std::endl;
  ProfileInfo profile_info = profiling_requests[guid];
  if (!output_filepath.empty()) {
    std::ofstream outputFile(output_filepath, std::ios::app);
    if (outputFile.is_open()) {
      outputFile << "Request " << guid << " profiling: " << std::endl;
      outputFile << "Decoding time: "
                  << (profile_info.finish_time -
                  profile_info.start_decoding_time) * 1e-3
                  << "ms" << std::endl;
      outputFile << "Total time: "
                  << (profile_info.finish_time -
                      profile_info.start_time) * 1e-3
                  << "ms" << std::endl;
      outputFile << "LLM decoding steps: "
                   << profile_info.llm_decoding_steps << std::endl;
      if (decoding_mode == SPECULATIVE_DECODING) {
        outputFile << "SSM decoding steps: "
                    << profile_info.ssm_decoding_steps << std::endl;
      }
      outputFile << output << std::endl << std::endl;
      outputFile.close();
    } else {
      std::cout << "Unable to open the output file: " << output_filepath
                << std::endl;
      assert(false);
    }
  }

  trigger_request_completion_future(guid);
}

void RequestManager::update_inference_results(InferenceResult const &result) {
  // Update the inference results
  std::lock_guard<std::mutex> const rm_state_lock(rm_state_mutex);
  std::lock_guard<std::mutex> const request_queue_lock(request_queue_mutex);

  if (num_available_requests == 0) {
    // Update nothing
    if (!pending_request_queue.empty()) {
      // Load the pending request to the batch
      load_pending_reqeust_to_batch();
      request_manager_status = PREFILLING;
      if (decoding_mode == SPECULATIVE_DECODING) {
        prefill_model = SSM;
      }
    }
    return;
  }

  switch (request_manager_status) {
    case PREFILLING:
      if (decoding_mode == INCREMENTAL_DECODING) {
        if (update_llm_prefill_results(result)) {
          // This indicates that the prefilling of the current request finishes
          // Reset the prefill_request
          prefill_request = nullptr;

          // Check if there are more empty slots
          if (num_available_requests < get_max_requests_per_batch() &&
              !pending_request_queue.empty()) {
            // Load the pending request to the batch
            load_pending_reqeust_to_batch();
            request_manager_status = PREFILLING;
          } else {
            // No more empty slots, start the decoding
            request_manager_status = DECODING;
          }
        }
        // Not completed, continue prefilling
      } else if (decoding_mode == SPECULATIVE_DECODING) {
        if (prefill_model == SSM) {
          if (update_ssm_prefill_results(result)) {
            // This indicates that the prefilling phase for SSM finishes
            // We need to start the LLM prefilling
            prefill_model = LLM;
          }
          // Not completed, continue SSM prefilling
        } else if (prefill_model == LLM) {
          if (update_llm_prefill_results(result)) {
            // This indicates that the prefilling phase finishes
            prefill_request = nullptr;
            // Check if there are more empty slots
            if (num_available_requests < get_max_requests_per_batch() &&
                !pending_request_queue.empty()) {
              // Load the pending request to the batch
              load_pending_reqeust_to_batch();
              request_manager_status = PREFILLING;
              prefill_model = SSM;
            } else {
              // No more empty slots, start the speculation
              request_manager_status = SSM_SPEC;
              // Reset the prefill_request
              current_speculation_step = 0;
            }
          }
          // Not completed, continue LLM prefilling
        } else {
          assert(false && "Invalid prefill model.");
        }
      } else {
        assert(false && "Invalid inference mode.");
      }
      break;
    case DECODING:
      if (update_llm_decode_results(result)) {
        // A request completed after the decode
        if (pending_request_queue.empty()) {
          // No pending request to process, continue the speculation
          request_manager_status = DECODING;
        } else {
          request_manager_status = PREFILLING;
          load_pending_reqeust_to_batch();
        }
      }
      break;
    case LLM_VERIFY:
      if (update_llm_verify_results(result)) {
        // A request completed after the verification
        if (pending_request_queue.empty()) {
          // No pending request to process, continue the speculation
          request_manager_status = SSM_SPEC;
          current_speculation_step = 0;
        } else {
          request_manager_status = PREFILLING;
          load_pending_reqeust_to_batch();
          prefill_model = SSM;
        }
      } else {
        request_manager_status = SSM_SPEC;
        current_speculation_step = 0;
      }
      break;
    case SSM_SPEC:
      if (update_ssm_inference_results(result)) {
        // Stop condition for the speculation phase has been reached
        request_manager_status = LLM_VERIFY;
      }
      // else, keep the current status
      break;
    default:
      assert(false && "Invalid request manager status.");
  }
}

bool RequestManager::update_llm_prefill_results(InferenceResult const &result) {
  bool prefill_completed = false;
  int committed_token_offset = prefill_request->llm_cache_size;
  prefill_request->llm_cache_size += prefill_request->num_tokens_in_batch;
  prefill_request->committed_tokens.clear();

  if (decoding_mode == SPECULATIVE_DECODING) {
    // Modified the state because the last commitment completes
    prefill_request->llm_committed = true;
    assert(prefill_request->ssm_committed and prefill_request->llm_committed);

    for (int i = 0; i < prefill_request->num_tokens_in_batch; i++) {
      prefill_request->committed_tokens.push_back(Request::CommittedToken{
          i,
          committed_token_offset + i,
          prefill_request->tokens[i + committed_token_offset]});
    }
    // Modified the state because the new commitment is unfinished
    prefill_request->llm_committed = false;
  }

  if (prefill_request->llm_cache_size == prefill_request->tokens.size()) {
    // Indicates that the LLM prefilling phase finishes
    prefill_request->tokens.push_back(
        result.token_ids[prefill_request->num_tokens_in_batch - 1]);
    prefill_completed = true;

    if (decoding_mode == SPECULATIVE_DECODING) {
      // Add the last token to the token tree
      prefill_request->committed_tokens.push_back(
          Request::CommittedToken{-1,
                                  (int)prefill_request->tokens.size() - 1,
                                  prefill_request->tokens.back()});
      // Modified the state because the ssm also need to commit the last token
      prefill_request->ssm_committed = false;

      init_token_tree(prefill_request->guid);
      add_root_to_spec_token_tree(prefill_request->guid,
                                  prefill_request->tokens.back());
      update_bitmask_prompt(prefill_request->guid, 1);
    }
  }

  profiling_requests[prefill_request->guid].llm_prefilling_steps++;

  // Manages the committed states for other requests in the batch
  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    if (request_index == prefill_request->batch_index) {

      continue;
    }

    if (!request.llm_committed) {
      request.llm_committed = true;
      if (request.ssm_committed and request.llm_committed) {
        request.llm_cache_size = request.tokens.size() - 1;
        request.committed_tokens.clear();
      }
    }
  }
  return prefill_completed;
}

bool RequestManager::update_llm_decode_results(InferenceResult const &result) {
  bool request_completed = false;
  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    request.llm_cache_size++;
    request.tokens.push_back(
        result.token_ids[request.first_token_offset_in_batch]);

    profiling_requests[guid].llm_decoding_steps++;
    if (request.tokens.size() >= get_max_sequence_length()) {
      request_completed = true;
      request_complete_clean_up(request_index);
    }

    if (verbose) {
      std::string output = this->tokenizer_->Decode(request.tokens);
      std::cout << "Request " << guid << " tokens: " << std::endl
                << output << std::endl;
    }
  }
  return request_completed;
}

bool RequestManager::update_ssm_prefill_results(
    InferenceResult const &ssm_prefill_result) {
  // This function is called by update_inference_results when the
  // request_manager_status is PREFILLING and the prefill_model is SSM.
  // There's no results to update, but we should update ssm_cache_size.
  prefill_request->ssm_cache_size += prefill_request->num_tokens_in_batch;

  profiling_requests[prefill_request->guid].ssm_prefilling_steps++;

  if (prefill_request->ssm_cache_size == prefill_request->tokens.size()) {
    return true;
  }
  return false;
}

BatchConfig RequestManager::prepare_next_batch() {
  switch (request_manager_status) {
    case PREFILLING:
      if (decoding_mode == INCREMENTAL_DECODING) {
        return prepare_llm_prefilling_batch();
      } else if (decoding_mode == SPECULATIVE_DECODING) {
        if (prefill_model == SSM) {
          return prepare_ssm_prefilling_batch();
        } else if (prefill_model == LLM) {
          return prepare_llm_prefilling_batch();
        } else {
          assert(false && "Invalid prefill model.");
        }
      } else {
        assert(false && "Invalid inference mode.");
      }
      break;
    case DECODING:
      return prepare_decoding_batch();
    case SSM_SPEC:
      if (current_speculation_step == 0) {
        return prepare_first_spec_batch_config();
      } else {
        return prepare_next_spec_batch_config();
      }
    case LLM_VERIFY:
      return prepare_verify_batch_config();
    default:
      std::cout << "Invalid request manager status: " << request_manager_status
                << std::endl;
      assert(false);
  }
}

BatchConfig RequestManager::prepare_llm_prefilling_batch() {
  // This function is called when the request_manager_status is PREFILLING,
  // which means that there is a request in the prefilling phase.
  // This function load its prefilling tokens, constructing a BatchConfig with
  // only one request.
  if (verbose) {
    std::cout << "\n############### prepare_llm_prefilling_batch "
                 "##############\n";
  }
  assert(prefill_request != nullptr &&
         "No prefilling request to process in the prefilling phase.");

  BatchConfig bc;
  if (decoding_mode == INCREMENTAL_DECODING) {
    bc.inference_mode = InferenceMode::INC_DECODING_MODE;
  } else if (decoding_mode == SPECULATIVE_DECODING) {
    bc.inference_mode = InferenceMode::TREE_VERIFY_MODE;
  }
  bc.prompt_phase = true;
  std::copy(std::begin(request_available),
            std::end(request_available),
            std::begin(bc.request_available));
  bc.num_available_requests = num_available_requests;

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    if (request_index == prefill_request->batch_index) {
      // Request Info
      bc.requestsInfo[request_index].first_token_offset_in_batch = 0;
      bc.requestsInfo[request_index].first_token_index_in_request =
          prefill_request->llm_cache_size;
      bc.requestsInfo[request_index].num_tokens_in_batch =
          std::min(get_max_tokens_per_batch(),
                   (int)prefill_request->tokens.size() -
                       prefill_request->llm_cache_size);

      prefill_request->first_token_offset_in_batch = 0;
      prefill_request->num_tokens_in_batch =
          bc.requestsInfo[request_index].num_tokens_in_batch;

      // Token Info
      for (int token_idx = 0;
           token_idx < bc.requestsInfo[request_index].num_tokens_in_batch;
           token_idx++) {
        int abs_idx = prefill_request->llm_cache_size + token_idx;
        assert(abs_idx < prefill_request->tokens.size());

        bc.tokensInfo[token_idx].request_index = request_index;
        bc.tokensInfo[token_idx].abs_index_in_request = abs_idx;
        bc.tokensInfo[token_idx].abs_depth_in_request = abs_idx;
        bc.tokensInfo[token_idx].token_id = prefill_request->tokens[abs_idx];

        bc.num_tokens++;
      }

    } else {
      bc.requestsInfo[request_index].first_token_offset_in_batch = 0;
      bc.requestsInfo[request_index].first_token_index_in_request =
          request.llm_cache_size;
      bc.requestsInfo[request_index].num_tokens_in_batch = 0;

      if (!request.llm_committed) {
        // Committed tokens
        for (int i = 0; i < request.committed_tokens.size() - 1; i++) {
          bc.committed_tokens[bc.num_tokens_to_commit].token_index =
              request.committed_tokens[i].from_index;
          bc.committed_tokens[bc.num_tokens_to_commit].request_index =
              request_index;
          bc.committed_tokens[bc.num_tokens_to_commit].token_depth =
              request.committed_tokens[i].to_index;
          bc.num_tokens_to_commit++;
        }
      }
    }
  }

  if (verbose) {
    std::cout << "prepare_llm_prefilling_batch NEW batchconfig:" << std::endl;
    bc.print();
  }
  return bc;
}

BatchConfig RequestManager::prepare_ssm_prefilling_batch() {
  // This function is called when the request_manager_status is PREFILLING,
  // which means that there is a request in the prefilling phase.
  // This function load its prefilling tokens, constructing a BatchConfig with
  // only one request.
  if (verbose) {
    std::cout << "\n############### prepare_ssm_prefilling_batch "
                 "##############\n";
  }
  assert(prefill_request != nullptr &&
         "No prefilling request to process in the prefilling phase.");

  BatchConfig bc;
  bc.inference_mode = InferenceMode::TREE_SEARCH_MODE;
  bc.prompt_phase = true;
  // Only set the prefilling request to be available
  bc.request_available[prefill_request->batch_index] = true;
  bc.num_available_requests = 1;

  int request_index = prefill_request->batch_index;
  // Request Info
  bc.requestsInfo[request_index].first_token_offset_in_batch = 0;
  bc.requestsInfo[request_index].first_token_index_in_request =
      prefill_request->ssm_cache_size;
  bc.requestsInfo[request_index].num_tokens_in_batch = std::min(
      get_max_tokens_per_batch(),
      (int)prefill_request->tokens.size() - prefill_request->ssm_cache_size);

  prefill_request->first_token_offset_in_batch = 0;
  prefill_request->num_tokens_in_batch =
      bc.requestsInfo[request_index].num_tokens_in_batch;

  // Token Info
  for (int token_idx = 0;
       token_idx < bc.requestsInfo[request_index].num_tokens_in_batch;
       token_idx++) {
    int abs_idx = prefill_request->ssm_cache_size + token_idx;
    assert(abs_idx < prefill_request->tokens.size());

    bc.tokensInfo[token_idx].request_index = request_index;
    bc.tokensInfo[token_idx].abs_index_in_request = abs_idx;
    bc.tokensInfo[token_idx].abs_depth_in_request = abs_idx;
    bc.tokensInfo[token_idx].token_id = prefill_request->tokens[abs_idx];

    bc.num_tokens++;
  }

  if (verbose) {
    std::cout << "prepare_ssm_prefilling_batch NEW batchconfig:" << std::endl;
    bc.print();
  }
  return bc;
}

BatchConfig RequestManager::prepare_decoding_batch() {
  // This function is called when the request_manager_status is DECODING. It
  // fills the last token of each request in the current batch to the
  // BatchConfig for the LLM to decode.
  if (verbose) {
    std::cout << "\n############### prepare_decoding_batch "
                 "##############\n";
  }

  BatchConfig bc;
  bc.inference_mode = InferenceMode::INC_DECODING_MODE;
  bc.prompt_phase = false;
  std::copy(std::begin(request_available),
            std::end(request_available),
            std::begin(bc.request_available));
  bc.num_available_requests = num_available_requests;

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       request_index++) {
    if (!request_available[request_index]) {
      continue;
    }
    Request &request = all_requests[guid_of_requests[request_index]];
    assert(request.status == Request::RUNNING);

    // Per Request Info
    bc.requestsInfo[request_index].first_token_index_in_request =
        request.llm_cache_size;
    bc.requestsInfo[request_index].first_token_offset_in_batch = bc.num_tokens;
    bc.requestsInfo[request_index].num_tokens_in_batch = 1;

    request.first_token_offset_in_batch = bc.num_tokens;
    request.num_tokens_in_batch = 1;

    // Per Token Info
    bc.tokensInfo[bc.num_tokens].request_index = request_index;
    bc.tokensInfo[bc.num_tokens].abs_index_in_request = request.llm_cache_size;
    bc.tokensInfo[bc.num_tokens].abs_depth_in_request = request.llm_cache_size;
    bc.tokensInfo[bc.num_tokens].token_id = request.tokens.back();

    bc.num_tokens++;

    if (profiling_requests[request.guid].llm_decoding_steps == 0) {
      profiling_requests[request.guid].start_decoding_time =
          Realm::Clock::current_time_in_microseconds();
    }
  }

  if (verbose) {
    std::cout << "prepare_decoding_batch NEW batchconfig:" << std::endl;
    bc.print();
  }
  return bc;
}
/* ----- Speculative Inference Specific functions ----- */

/***** Request Init Phase *****/
BatchConfig RequestManager::prepare_first_spec_batch_config() {
  if (verbose) {
    std::cout << "\n############### prepare_first_spec_batch_config "
                 "##############\n";
  }
  // This method does the following:
  // 1. Commit the verified tokens through BatchConfig. The infomation
  // of the committed tokens are stored in request.committed_tokens. Put the
  // information of the committed tokens into BatchConfig.TokensInfo.
  // 2. Maintain BatchConfig::RequestsInfo and all other fields of
  // BatchConfig.
  assert(current_speculation_step == 0);

  BatchConfig new_bc;
  new_bc.inference_mode = InferenceMode::TREE_SEARCH_MODE;
  // Assume that only one small model is in use now
  new_bc.prompt_phase = true;
  std::copy(std::begin(request_available),
            std::end(request_available),
            std::begin(new_bc.request_available));
  new_bc.num_available_requests = num_available_requests;

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    std::vector<Request::CommittedToken> &committed_tokens =
        request.committed_tokens;

    // Maintain requestsInfo
    new_bc.requestsInfo[request_index].first_token_offset_in_batch =
        new_bc.num_tokens;
    new_bc.requestsInfo[request_index].first_token_index_in_request =
        request.ssm_cache_size;
    // We don't directly use committed_tokens.size() here because there is a
    // case where committed_tokens.size() != request.tokens.size() -
    // request.ssm_cache_size, that's when the LLM prefilling is just finished
    new_bc.requestsInfo[request_index].num_tokens_in_batch =
        request.tokens.size() - request.ssm_cache_size;

    request.first_token_offset_in_batch = new_bc.num_tokens;
    request.num_tokens_in_batch =
        request.tokens.size() - request.ssm_cache_size;

    // Store committed tokens to tokensInfo
    int start_offset = committed_tokens.size() - request.tokens.size() +
                       request.ssm_cache_size;
    assert(start_offset >= 0 && "Invalid start offset.");
    for (int committed_token_index = start_offset;
         committed_token_index < committed_tokens.size();
         committed_token_index++) {
      new_bc.tokensInfo[new_bc.num_tokens].request_index = request_index;
      new_bc.tokensInfo[new_bc.num_tokens].abs_index_in_request =
          committed_tokens[committed_token_index].to_index;
      new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
          committed_tokens[committed_token_index].to_index;
      new_bc.tokensInfo[new_bc.num_tokens].token_id =
          committed_tokens[committed_token_index].token_id;
      new_bc.num_tokens++;
    }

    // Copy the causal mask, it should already been updated in
    // update_llm_verify_results
    new_bc.causalMask[request_index] = request.causal_mask;

    if (profiling_requests[guid].ssm_decoding_steps == 0) {
      profiling_requests[guid].start_decoding_time =
          Realm::Clock::current_time_in_microseconds();
    }
  }
  if (verbose) {
    std::cout << "prepare_first_spec_batch_config NEW batchconfig:"
              << std::endl;
    new_bc.print();
  }
  return new_bc;
}

/***** Speculative Decoding Phase *****/
BatchConfig RequestManager::prepare_next_spec_batch_config() {
  if (verbose) {
    std::cout
        << "\n############### prepare_next_spec_batch_config ###############\n";
    std::cout << "Current tree depth: " << current_speculation_step + 1 << "\n";
  }

  // Prepare the next batch for existing requests
  BatchConfig new_bc;
  new_bc.inference_mode = InferenceMode::TREE_SEARCH_MODE;
  // We assume that only one small model is in use now
  new_bc.model_id = 0;
  std::copy(std::begin(request_available),
            std::end(request_available),
            std::begin(new_bc.request_available));
  new_bc.num_available_requests = num_available_requests;

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    new_bc.requestsInfo[request_index].first_token_offset_in_batch =
        new_bc.num_tokens;

    // Fill in the tokens
    TokenTree &token_tree = request.speculative_token_trees.at(new_bc.model_id);
    if (token_tree.tree_layers.size() <= current_speculation_step) {
      // This request has no token to decode in this and the following small
      // model inference steps
      new_bc.requestsInfo[request_index].num_tokens_in_batch = 0;
      new_bc.requestsInfo[request_index].first_token_index_in_request =
          request.causal_mask.non_tree_cache_size +
          request.causal_mask.tree_or_prompt_size -
          request.causal_mask.current_layer_size;
      request.num_tokens_in_batch = 0;
      request.first_token_offset_in_batch = new_bc.num_tokens;
      continue;
    } else {
      std::list<std::shared_ptr<TokenTreeNode>> &current_layer =
          token_tree.tree_layers.back();
      // Exclude the current layer from the token tree, because we want the
      // start index
      new_bc.requestsInfo[request_index].first_token_index_in_request =
          request.causal_mask.non_tree_cache_size +
          request.causal_mask.tree_or_prompt_size -
          request.causal_mask.current_layer_size;
      new_bc.requestsInfo[request_index].num_tokens_in_batch =
          request.causal_mask.current_layer_size;

      request.num_tokens_in_batch =
          new_bc.requestsInfo[request_index].num_tokens_in_batch;
      request.first_token_offset_in_batch = new_bc.num_tokens;

      int child_index = 0;
      for (auto const &node_ptr : current_layer) {
        new_bc.tokensInfo[new_bc.num_tokens].request_index = request_index;
        new_bc.tokensInfo[new_bc.num_tokens].abs_index_in_request =
            new_bc.requestsInfo[request_index].first_token_index_in_request +
            child_index;
        new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
            request.tokens.size() - 1 + current_speculation_step;
        new_bc.tokensInfo[new_bc.num_tokens].token_id = node_ptr->id;

        new_bc.num_tokens++;
        child_index++;
      }
    }

    // Copy the causal mask, it should already been updated by
    // update_ssm_inference_results
    new_bc.causalMask[request_index] = request.causal_mask;
  }

  if (verbose) {
    std::cout << "prepare_next_spec_batch_config NEW batchconfig:" << std::endl;
    new_bc.print();
  }
  return new_bc;
}

/***** Verify Phase *****/
BatchConfig RequestManager::prepare_verify_batch_config() {
  if (verbose) {
    std::cout
        << "\n############### prepare_verify_batch_config ###############\n";
  }
  // This method does the following:
  // 1. Commit the verified tokens in the last iteration through the
  // BatchConfig. We can do this request by request.
  // The information of the committed tokens is stored in
  // Request.llm_committed_tokens. Put the information of the committed tokens
  // into BatchConfig.committed_tokens.
  // 2. Load the tokens on the token tree that are not yet pruned to
  // BatchConfig.tokensInfo. Be careful with the abs_depth etc.
  // (skip the pruned tokens).
  // 3. Create the causal mask for the large model based on the small model
  // causal mask (call create_llm_bitmask()).
  // 4. Maintain BatchConfig::RequestsInfo and all other fields of
  // BatchConfig.
  // Please refer to the implementation of prepare_next_spec_batch_config()
  // for more details.
  BatchConfig new_bc;
  new_bc.inference_mode = InferenceMode::TREE_VERIFY_MODE;
  std::copy(std::begin(request_available),
            std::end(request_available),
            std::begin(new_bc.request_available));
  new_bc.num_available_requests = num_available_requests;

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    // 1. Maintain requestsInfo
    new_bc.requestsInfo[request_index].first_token_index_in_request =
        request.tokens.size() - 1; // Exclude the last token
    new_bc.requestsInfo[request_index].first_token_offset_in_batch =
        new_bc.num_tokens;
    new_bc.requestsInfo[request_index].num_tokens_in_batch = 0;

    // Put the information of the committed tokens into
    // BatchConfig.committed_tokens.
    // Note here, we shouldn't put the last token in request.committed_tokens
    // into new_bc. Because the LLM don't have that token's KV cache.
    if (!request.llm_committed) {
      std::vector<Request::CommittedToken> &committed_tokens =
          request.committed_tokens;
      for (int committed_token_index = 0;
           committed_token_index < committed_tokens.size() - 1;
           committed_token_index++) {
        Request::CommittedToken &committed_token =
            committed_tokens.at(committed_token_index);
        new_bc.committed_tokens[new_bc.num_tokens_to_commit].request_index =
            request_index;
        new_bc.committed_tokens[new_bc.num_tokens_to_commit].token_index =
            committed_token.from_index;
        new_bc.committed_tokens[new_bc.num_tokens_to_commit].token_depth =
            committed_token.to_index;
        new_bc.num_tokens_to_commit++;
      }
    }

    // Load the tokens on the token tree that are not yet pruned to
    // BatchConfig.tokensInfo.
    TokenTree &token_tree = request.speculative_token_trees[0];
    int token_tree_index = 0;
    int layer_index = 0;
    for (auto const &tree_layer : token_tree.tree_layers) {
      for (auto const &tree_node : tree_layer) {
        if (tree_node->pruned == false) {
          new_bc.tokensInfo[new_bc.num_tokens].request_index = request_index;
          new_bc.tokensInfo[new_bc.num_tokens].abs_index_in_request =
              request.tokens.size() - 1 + token_tree_index;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
              request.tokens.size() - 1 + layer_index;
          new_bc.tokensInfo[new_bc.num_tokens].token_id = tree_node->id;
          new_bc.num_tokens++;
          token_tree_index++;
        }
      }
      layer_index++;
    }
    assert(token_tree_index == token_tree.tree_size);
    new_bc.requestsInfo[request_index].num_tokens_in_batch = token_tree_index;

    request.first_token_offset_in_batch = new_bc.num_tokens - token_tree_index;
    request.num_tokens_in_batch = token_tree_index;

    // Create the causal mask for the large model based on the small model
    // causal mask.
    new_bc.causalMask[request_index] = create_llm_bitmask(guid);
  }

  if (verbose) {
    std::cout << "prepare_verify_batch_config NEW batchconfig:" << std::endl;
    new_bc.print();
  }
  return new_bc;
}

bool RequestManager::update_llm_verify_results(
    InferenceResult const &llm_verify_result) {
  // We may have two types of InferenceResults, one is the results from
  // sampling the large model, the other is the top-p / top-k logits of the
  // large model, we can first implement the former one. For the latter one,
  // we have to add a CPU based verify function.

  // Compare the results returned from the LLM and compare them with the
  // SSM's speculative token tree. For the greedy construction of the
  // speculative token tree, we can simply compare LLM's sample result at each
  // token, this is implemented in get_verify_results_greedy(). This function
  // stores the commmitted tokens into the corresponding fields in the
  // Request. For the sampling construction of the speculative token tree, we
  // need to implement a CPU based verify function.

  // Update llm_cache_size with the last committed_tokens, and clear
  // committed_tokens
  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    if (!request.llm_committed) {
      request.llm_committed = true;
      request.llm_cache_size +=
          request.committed_tokens.size() - 1; // Exclude the last token
      // Check if both the KV cache of SSM and LLM are committed, because
      // sometimes the LLM KV cache is committed by a verifying batch config,
      // sometimes it is committed by a LLM prefilling batch config. We don't
      // know when the tokens are committed, so we have to add these checks
      // whenever the SSM or the LLM commits tokens. If the both caches are
      // committed, we can clear the committed tokens.
      if (request.ssm_committed and request.llm_committed) {
        request.committed_tokens.clear();
      }
    }

    profiling_requests[guid].llm_decoding_steps++;
  }

  // Process the LLM results greedily
  get_verify_results_greedy(llm_verify_result);

  // Clear the token tree node pool
  token_tree_node_pool = std::priority_queue<
      std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid>,
      std::vector<std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid>>,
      CompareSharedTokenTreeNodePtrRequestGuidPair>();

  bool request_completed = false;

  // Iterate over the requests
  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    if (verbose) {
      std::cout << "Request " << guid << " token tree: " << std::endl;
      std::cout << request.speculative_token_trees[0];
    }

    // Initialize the token tree for the request
    init_token_tree(guid);
    assert(!request.committed_tokens.empty() &&
           "The committed tokens should not be empty.");
    // Add the last committed token as the root of the speculative token tree
    add_root_to_spec_token_tree(guid, request.committed_tokens.back().token_id);

    // Check if the request is completed. If its completed, clean up the
    // metainfo stored in the RequestManager. Otherwise, update its bitmask.
    if (request.tokens.size() >= get_max_sequence_length()) {
      // Request is completed
      request_completed = true;
      request_complete_clean_up(request_index);
    } else {
      update_bitmask_prompt(guid, request.committed_tokens.size() - 1);
    }
  }

  // Some requests may be completed after appending the verified tokens.
  // If there is a request completed, return true.
  return request_completed;
}

bool RequestManager::update_ssm_inference_results(
    InferenceResult const &ssm_inference_result) {
  // This function returns false if no tokens are added to the token tree,
  // which indicates that the ssm inference phase is done.
  assert(current_speculation_step >= 0 &&
         "The current speculation step should be no less than 0");
  current_speculation_step++;

  int num_branches = BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
  int result_index = 0;

  // Here we assume that the order of the tokens in the last
  // BatchConfig and hence the last InferenceResult is equal to
  // the order of the request in the last BatchConfig
  bool all_request_last_layer_empty =
      add_tokens_to_spec_token_tree(ssm_inference_result);

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    if (current_speculation_step == 1) {
      request.ssm_committed = true;
      // Check if both the KV cache of SSM and LLM are committed, because
      // sometimes the LLM KV cache is committed by a verifying batch config,
      // sometimes it is committed by a LLM prefilling batch config. We don't
      // know when the tokens are committed, so we have to add these checks
      // whenever the SSM or the LLM commits tokens. If the both caches are
      // committed, we can clear the committed tokens.
      if (request.ssm_committed and request.llm_committed) {
        request.committed_tokens.clear();
      }
      request.ssm_cache_size = request.tokens.size();
    }

    if (current_speculation_step == 1) {
      init_bitmask_spec(guid);
    }
    append_bitmask(guid);

    profiling_requests[guid].ssm_decoding_steps++;
  }

  // Stop conditions
  return all_request_last_layer_empty ||
         current_speculation_step > get_max_tree_depth();
}

/* --------- Bitmask Related Functions --------- */

void RequestManager::init_bitmask_prompt(RequestGuid guid, int prompt_length) {
  // This method is called by load_pending_reqeust_to_batch when there is a
  // new request to load into the batch
  Request &request = all_requests[guid];
  BatchConfig::BitMask &bitmask = request.causal_mask;

  // Clear because the prompt kernel doesn't use mask
  bitmask.clear_bitmask();
  // Set the info for the mask which is used to store the KV cache
  bitmask.tree_or_prompt_size = prompt_length;
  bitmask.current_layer_size = prompt_length;
  bitmask.non_tree_cache_size = 0;
}

void RequestManager::update_bitmask_prompt(RequestGuid guid,
                                           int num_committed_tokens) {
  // This method modifies the bitmask in place
  // This method is called by update_llm_verify_results
  // 1. Clear the causal mask because the first SSM inference uses the prompt
  // kernel and it doesn't use mask.
  // 2. Maintain all other fields.
  Request &request = all_requests[guid];
  BatchConfig::BitMask &bitmask = request.causal_mask;
  // Clear because the prompt kernel doesn't use mask
  bitmask.clear_bitmask();
  bitmask.tree_or_prompt_size = num_committed_tokens;
  bitmask.current_layer_size = num_committed_tokens;

  // If the request just finishes the prefilling phase, we need to set the
  // non_tree_cache_size to the size of the prompt
  if (bitmask.non_tree_cache_size == 0) {
    bitmask.non_tree_cache_size = request.tokens.size() - num_committed_tokens;
  }
}

void RequestManager::init_bitmask_spec(RequestGuid guid) {
  // This method modifies the bitmask in place
  // This method is called by the first call of update_ssm_inference_results
  // in a speculative iteration CAUTION: You should still call
  // append_bitmask() after this method
  // 1. Clear the causal mask and add a root into it, because the tree is
  // currently empty but we have a root.
  // 2. Maintain all other fields.
  assert(current_speculation_step == 1 &&
         "The current speculation step should be 1");
  Request &request = all_requests[guid];
  request.causal_mask = BatchConfig::BitMask();
  // Set the mask for the root
  request.causal_mask.bit_mask[0].set_bit(0);
  request.causal_mask.tree_or_prompt_size = 1;
  request.causal_mask.non_tree_cache_size = request.tokens.size() - 1;
  request.causal_mask.current_layer_size = 1;
}

void RequestManager::append_bitmask(RequestGuid guid) {
  // This method changes the bitmask in place
  // This method is called by update_ssm_inference_results(), after the new
  // tokens are added to the token tree
  assert(current_speculation_step >= 1 &&
         "The current speculation step should be no less than 1");

  Request &request = all_requests[guid];
  BatchConfig::BitMask &bitmask = request.causal_mask;
  TokenTree &token_tree = request.speculative_token_trees[0];

  if (token_tree.tree_layers.size() <= current_speculation_step) {
    // This request has no token added in this and the following small model
    // inference steps, skip it
    return;
  }
  std::list<std::shared_ptr<TokenTreeNode>> &tree_layer =
      request.speculative_token_trees[0].tree_layers.back();
  int new_layer_size = tree_layer.size();
  int last_layer_size = bitmask.current_layer_size;
  int previous_tree_size = bitmask.tree_or_prompt_size;
  bitmask.current_layer_size = new_layer_size;
  bitmask.tree_or_prompt_size += new_layer_size;

  assert(bitmask.tree_or_prompt_size <= get_max_spec_tree_token_num());

  int parent_offset = previous_tree_size - last_layer_size;
  int child_offset = previous_tree_size;

  int child_idx = 0;
  for (auto const &child_ptr : tree_layer) {
    // Each child copy its parent's mask
    bitmask.bit_mask[child_offset + child_idx] =
        bitmask.bit_mask[parent_offset + child_ptr->parent_pos];
    // Each child attend to itself
    bitmask.bit_mask[child_offset + child_idx].set_bit(child_offset +
                                                       child_idx);
    child_idx++;
  }
}

BatchConfig::BitMask RequestManager::create_llm_bitmask(RequestGuid guid) {
  // This method creates a new bitmask for LLM verification model's bitmask,
  // it does not modify the small model's bitmask This method is called by
  // prepare_verify_batch_config().

  Request &request = all_requests[guid];
  TokenTree &token_tree = request.speculative_token_trees[0];
  BatchConfig::BitMask llm_bitmask = BatchConfig::BitMask();

  int abs_index_in_tree = 0;
  std::vector<int> parent_pos_2_abs_index;
  std::vector<int> current_layer_abs_index;
  for (auto const &tree_layer : token_tree.tree_layers) {
    for (auto const &tree_node : tree_layer) {
      current_layer_abs_index.push_back(abs_index_in_tree);
      if (tree_node->pruned == false) {
        if (abs_index_in_tree == 0) {
          // The root token, set itself
          llm_bitmask.bit_mask[0].set_bit(0);
        } else {
          // Copy from the parent, and set itself
          int parent_abs_index = parent_pos_2_abs_index[tree_node->parent_pos];
          llm_bitmask.bit_mask[abs_index_in_tree] =
              llm_bitmask.bit_mask[parent_abs_index];
          llm_bitmask.bit_mask[abs_index_in_tree].set_bit(abs_index_in_tree);
        }
        abs_index_in_tree++;
      }
    }
    parent_pos_2_abs_index.clear();
    parent_pos_2_abs_index.swap(current_layer_abs_index);
  }

  // A sanity check
  assert(abs_index_in_tree == token_tree.tree_size);

  // Maintain other fields of llm_bitmask
  llm_bitmask.non_tree_cache_size = request.causal_mask.non_tree_cache_size;
  // We don't need to set llm_bitmask.current_layer_size and
  // llm_bitmask.tree_or_prompt_size here because they are not used in LLM
  // verification.
  return llm_bitmask;
}
/* --------- Bitmask Related Functions --------- */

void RequestManager::get_verify_results_greedy(
    InferenceResult const &llm_verify_result) {
  // This function maintain the generated token list of the request and the
  // committed tokens.
  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    int llm_result_offset = request.first_token_offset_in_batch;
    int committed_token_index = request.tokens.size() - 1;

    TokenTree &token_tree = request.speculative_token_trees[0];
    // First add the root to the committed tokens
    request.committed_tokens.push_back(Request::CommittedToken(
        llm_result_offset, committed_token_index, request.tokens.back()));
    committed_token_index++;
    // Don't add it to request.tokens because it has already been added.

    // The position of the last accepted token in its tree layer (includeing
    // the pruned tokens)
    int last_accepted_token_index_in_layer = 0;
    // The index of the last accepted token in the entire tree (excluding the
    // pruned tokens)
    int last_accepted_token_index = 0;

    int current_token_index = 1; // Because we skip the root
    auto layer_it = token_tree.tree_layers.begin();
    ++layer_it;
    for (; layer_it != token_tree.tree_layers.end(); ++layer_it) {
      // We skip the first layer
      std::list<std::shared_ptr<TokenTreeNode>> const &tree_layer = *layer_it;

      bool token_accepted_this_layer = false;
      int current_token_index_in_layer = 0;

      for (auto const &node_ptr : tree_layer) {
        if (node_ptr->pruned) {
          current_token_index_in_layer++;
          continue;
        }
        if ((node_ptr->parent_pos != last_accepted_token_index_in_layer) ||
            token_accepted_this_layer) {
          // The token's parent is not accepted, or there is already another
          // token accepted in this layer
          current_token_index++;
          current_token_index_in_layer++;
          continue;
        } else {
          // The token's parent is accepted, and no token has been accepted in
          // this layer yet
          if (node_ptr->id ==
              llm_verify_result
                  .token_ids[llm_result_offset + last_accepted_token_index]) {
            // The token's parent is accepted, and this token's id equals the
            // llm's sample at its parent's position. We accept this token.

            // from_index: the index of the token in the tree (excluding the
            // pruned tokens)
            // to_index: the committed token index in the request
            request.committed_tokens.push_back(
                Request::CommittedToken(llm_result_offset + current_token_index,
                                        committed_token_index,
                                        node_ptr->id));
            request.tokens.push_back(node_ptr->id);

            token_accepted_this_layer = true;
            last_accepted_token_index = current_token_index;
            last_accepted_token_index_in_layer = current_token_index_in_layer;
            committed_token_index++;
          }
          current_token_index++;
          current_token_index_in_layer++;
        }
      }
      if (!token_accepted_this_layer) {
        // No token is accepted in this layer, we should stop the traversal
        break;
      }
    }

    // Add the last token (that is not verified by the LLM)
    // from_index: since this token is not in the token tree, the llm
    // doesn't have its KV cache, so the from_index should be a place
    // holder, which is -1
    request.committed_tokens.push_back(Request::CommittedToken(
        -1,
        committed_token_index,
        llm_verify_result
            .token_ids[llm_result_offset + last_accepted_token_index]));
    request.tokens.push_back(
        llm_verify_result
            .token_ids[llm_result_offset + last_accepted_token_index]);

    request.llm_committed = false;
    request.ssm_committed = false;

    if (verbose) {
      std::cout << "Request " << request.guid << " committed tokens: ";
      for (auto const &committed_token : request.committed_tokens) {
        std::cout << committed_token.token_id << " ("
                  << tokenizer_->Decode({committed_token.token_id}) << ") ";
      }
      std::cout << std::endl;
      std::string output = this->tokenizer_->Decode(request.tokens);
      std::cout << "Output sequence: " << output << std::endl;
    }
  }
}

// TODO: the max_seq_length is not used in the current implementation
std::vector<GenerationResult>
    FFModel::generate(std::vector<std::string> &prompts, int max_seq_length) {
  RequestManager *rm = RequestManager::get_request_manager();
  std::vector<RequestManager::RequestGuid> guids;
  for (int i = 0; i < prompts.size(); i++) {
    RequestManager::RequestGuid guid = rm->register_new_request(prompts.at(i));
    if (guid != RequestManager::INVALID_GUID) {
      guids.push_back(guid);
    }
  }
  std::vector<GenerationResult> results;
  for (int i = 0; i < guids.size(); i++) {
    results.push_back(rm->get_generation_result(guids[i]));
  }
  return results;
}

void RequestManager::start_background_server(FFModel *model) {
  assert(background_server_status == INITIALIZED);
  background_server_status = SERVING;
  // Start background task
  Runtime *runtime = Runtime::get_runtime();
  Context ctx = Runtime::get_context();
  TaskLauncher launcher(RM_BACKGROUND_SERVING_TASK_ID,
                        TaskArgument(&model, sizeof(FFModel *)));
  background_server_handler = runtime->execute_task(ctx, launcher);
  // Register callbacks for normal exit
  {
    int ret = std::atexit(RequestManager::terminate_background_server_at_exit);
    assert(ret == 0); // make sure the callback is successfully registered
  }
  // Register callbacks for termination
  {
    std::set_terminate([]() {
      RequestManager::terminate_background_server_at_exit();
      std::abort();
    });
  }
}

void RequestManager::background_serving_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  RequestManager *rm = RequestManager::get_request_manager();
  FFModel *llm = *(FFModel **)task->args;
  {
    // Update FFModel's lg_hlr and lg_ctx to the current
    // task's runtime and ctx, since all future legion tasks are
    // launched in this task
    llm->config.lg_hlr = runtime;
    llm->config.lg_ctx = ctx;
    // Update the lg_hlr and lg_ctx of all SSMs' FFConfig
    // since all future legion tasks are launched in this task
    for (size_t i = 0; i < rm->get_num_ssms(); i++) {
      FFModel *ssm = rm->get_ssm_model(i);
      ssm->config.lg_hlr = runtime;
      ssm->config.lg_ctx = ctx;
    }
  }
  if (rm->decoding_mode == INCREMENTAL_DECODING) {
    // No SSMs: perform incremental decoding
    rm->serve_decoding(llm);
  } else {
    // Registered SSMs: perform speculative inference
    rm->serve_spec_infer(llm);
  }
}

/*static*/
void RequestManager::serve_decoding(FFModel *llm) {
  Context ctx = llm->config.lg_ctx;
  Runtime *runtime = llm->config.lg_hlr;
  // Compile the llm
  InferenceManager *im = InferenceManager::get_inference_manager();
  im->compile_model_and_allocate_buffer(llm);
  assert(im->model_weights_loaders.find(llm) !=
         im->model_weights_loaders.end());
  // Load model weights
  im->model_weights_loaders[llm]->load_weights(llm);
  // init operators
  im->init_operators_inference(llm);
  // Legion futures for inc_decoding and spec_infer
  InferenceResultFuture last_irf;
  {
    // Initialize futures for incr decoding
    InferenceResult ir;
    last_irf = Future::from_value<InferenceResult>(ir);
  }

  std::queue<InferenceResultFuture> batch_pipeline;
  { batch_pipeline.push(last_irf); }

  while (!is_background_server_terminated()) {

    if (batch_pipeline.size() >= 4) {
      // Block here to avoid launching too many batches
      auto const &ir = batch_pipeline.front();
      ir.get_void_result();
    }
    // deque finished batches
    while (batch_pipeline.size() > 1) {
      auto const &ir = batch_pipeline.front();
      if (ir.is_ready()) {
        batch_pipeline.pop();
      } else {
        break;
      }
    }
    runtime->begin_trace(ctx, 12346 /*trace_id*/);
    InferenceResultFuture next_ir = batch_pipeline.back();
    BatchConfigFuture bcf = get_next_batch_config(next_ir, ctx, runtime);
    FutureMap fm = im->inference(llm, 0, bcf);
    assert(fm.get_future_map_domain().get_volume() == 1);
    InferenceResultFuture irf = fm.get_future(0);
    batch_pipeline.push(irf);
    last_irf = irf;
    runtime->end_trace(ctx, 12346 /*trace_id*/);
  }
}

/*static*/
void RequestManager::serve_spec_infer(FFModel *llm) {
  Context ctx = llm->config.lg_ctx;
  Runtime *runtime = llm->config.lg_hlr;
  InferenceManager *im = InferenceManager::get_inference_manager();
  {
    // Compile the llm
    im->compile_model_and_allocate_buffer(llm);
    assert(im->model_weights_loaders.find(llm) !=
           im->model_weights_loaders.end());
    // Load model weights
    im->model_weights_loaders[llm]->load_weights(llm);
    // init operators
    im->init_operators_inference(llm);
  }
  for (size_t i = 0; i < get_num_ssms(); i++) {
    // Compile the i-th ssm
    FFModel *ssm = get_ssm_model(i);
    im->compile_model_and_allocate_buffer(ssm);
    assert(im->model_weights_loaders.find(ssm) !=
           im->model_weights_loaders.end());
    // Load model weights
    im->model_weights_loaders[ssm]->load_weights(ssm);
    // init operators
    im->init_operators_inference(ssm);
  }

  InferenceResultFuture last_irf;
  {
    // Initialize futures for incr decoding
    InferenceResult ir;
    last_irf = Future::from_value<InferenceResult>(ir);
  }

  request_manager_status = PREFILLING;
  prefill_model = SSM;

  long long time_1 = Realm::Clock::current_time_in_microseconds(), time_2;

  while (!is_background_server_terminated()) {
    // last_irf.get_void_result();
    BatchConfigFuture bcf = get_next_batch_config(last_irf, ctx, runtime);
    bcf.get_void_result();
    time_2 = Realm::Clock::current_time_in_microseconds();
    std::cout << "Iteration time: " << (time_2 - time_1) * 1e-3 << "ms"
              << std::endl;

    time_1 = Realm::Clock::current_time_in_microseconds();
    if ((request_manager_status == PREFILLING and prefill_model == LLM) or
        request_manager_status == LLM_VERIFY) {
      //   std::cout << "Branch 1" << std::endl;
      runtime->begin_trace(ctx, 12345 /*trace_id*/);
      FutureMap fm = im->inference(llm, 0, bcf);
      //   assert(fm.get_future_map_domain().get_volume() == 1);
      last_irf = fm.get_future(0);
      runtime->end_trace(ctx, 12345 /*trace_id*/);
    } else if ((request_manager_status == PREFILLING and
                prefill_model == SSM) or
               request_manager_status == SSM_SPEC) {
      //   std::cout << "Branch 2" << std::endl;
      runtime->begin_trace(ctx, 23456 /*trace_id*/);
      FutureMap fm = im->inference(get_ssm_model(0), 0, bcf);
      //   assert(fm.get_future_map_domain().get_volume() == 1);
      last_irf = fm.get_future(0);
      runtime->end_trace(ctx, 23456 /*trace_id*/);
    } else {
      assert(false && "Invalid request manager status");
    }
  }
}

void RequestManager::trigger_request_completion_future(
    RequestGuid const &guid) {
  std::lock_guard<std::mutex> const lock(request_to_promise_mutex);
  assert(request_to_promise.find(guid) != request_to_promise.end());
  // Set the completion promise in case other threads are waiting
  request_to_promise[guid]->set_value();
}

/*static*/
void RequestManager::terminate_background_server_at_exit() {
  RequestManager *rm = RequestManager::get_request_manager();
  rm->terminate_background_server();
}

void RequestManager::terminate_background_server() {
  if (background_server_status == SERVING) {
    background_server_status = TERMINATED;
    // Wait for the background server to terminate
    Runtime *runtime = Runtime::get_runtime();
    Context ctx = Runtime::get_context();
    background_server_handler.get_void_result();
  }
}

bool RequestManager::is_background_server_terminated() {
  return background_server_status == TERMINATED;
}

RequestManager *request_manager_singleton = nullptr;

/*static*/
RequestManager *RequestManager::get_request_manager() {
  if (request_manager_singleton == nullptr) {
    request_manager_singleton = new RequestManager();
  }
  return request_manager_singleton;
}

/* --------- Request Token Tree Related Functions --------- */
void RequestManager::init_token_tree(RequestGuid guid) {
  Request &request = all_requests[guid];
  request.speculative_token_trees.clear();
  // Assume we only use one small model for speculation
  request.speculative_token_trees.emplace_back();
}

void RequestManager::add_root_to_spec_token_tree(
    RequestGuid guid, BatchConfig::TokenId token_id) {
  // This method is called by update_llm_verify_results()
  // The last token in the accepted sequence should be the root of the next
  // speculation tree. The reason is that the KV cache of this token is not
  // computed yet, and we need the large model to decode the logit of this
  // token to verify its childs (the tokens in the first layer). This method
  // should: construct and add the root token to the empty speculative token
  // tree, with parent_pos being -1 and log_accumulated_prob being 0.0
  Request &request = all_requests[guid];
  TokenTree &speculative_token_tree = request.speculative_token_trees[0];
  speculative_token_tree.add_layer();
  auto node_ptr = std::make_shared<TokenTreeNode>(token_id, 0.0, -1);
  token_tree_node_pool.push(std::make_pair(node_ptr, guid));
  speculative_token_tree.tree_layers.front().push_back(node_ptr);
  speculative_token_tree.tree_size++;
}

bool RequestManager::add_tokens_to_spec_token_tree(
    InferenceResult const &ssm_inference_result) {

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);

    int parent_num = request.num_tokens_in_batch;
    if (parent_num == 0) {
      // The request has no committed tokens, we don't need to add tokens to the
      // token tree
      continue;
    }
    int result_offset = request.first_token_offset_in_batch *
                        BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
    int current_tree_size = request.causal_mask.tree_or_prompt_size;
    int empty_slots_in_layer =
        min(get_max_spec_tree_token_num() - current_tree_size,
            get_max_tree_width()); // The number of empty slots

    if (empty_slots_in_layer == 0) {
      // The token tree is full, we don't need to add tokens to it
      continue;
    }

    bool token_pool_full =
        token_tree_node_pool.size() >= get_max_tokens_per_batch();

    TokenTree &spec_token_tree = request.speculative_token_trees[0];
    std::list<std::shared_ptr<TokenTreeNode>> &last_layer =
        spec_token_tree.tree_layers.back();
    std::set<std::shared_ptr<TokenTreeNode>, CompareSharedTokenTreeNodePtr>
        tokens;
    int parent_pos = 0;
    for (auto const &parent_ptr : last_layer) {
      if (!parent_ptr->pruned) {
        for (int child_pos = 0;
             child_pos < BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
             child_pos++) {
          int result_idx =
              result_offset +
              parent_pos * BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES +
              child_pos;
          float log_prob = log(ssm_inference_result.probs[result_idx]);
          float log_accumulated_prob =
              log_prob + parent_ptr->log_accumulated_prob;

          //   std::cout << "Probability at result index " << result_idx << ": "
          //             << ssm_inference_result.probs[result_idx] << "\t";
          //   std::cout << "Token id: "
          //             << ssm_inference_result.token_ids[result_idx] <<
          //             std::endl;
          assert(log_prob != -std::numeric_limits<float>::infinity() &&
                 "Child log probability should not be -inf.");

          if (tokens.size() == empty_slots_in_layer and
              log_accumulated_prob <= (*tokens.begin())->log_accumulated_prob) {
            // The token tree is full, and the new token has a lower joint
            // probability than the minimum node in the pool, we don't need to
            // add the new token and the following tokens belong to the same
            // parent to the tree, because the tokens are sorted by their
            // probability
            break;
          } else if (token_pool_full and
                     log_accumulated_prob <= token_tree_node_pool.top()
                                                 .first->log_accumulated_prob) {
            // The token tree is not full, but the token pool is full, and the
            // new token has a lower joint probability than the minimum node in
            // the pool, we don't need to add the new token and the following
            // tokens belong to the same parent to the tree, because the tokens
            // are sorted by their probability
            break;
          } else {
            std::shared_ptr<TokenTreeNode> node_ptr =
                std::make_shared<TokenTreeNode>(
                    ssm_inference_result.token_ids[result_idx],
                    log_accumulated_prob,
                    parent_pos);
            if (tokens.size() == empty_slots_in_layer and
                log_accumulated_prob >
                    (*tokens.begin())->log_accumulated_prob) {
              // The token tree is full, and the new token has a higher joint
              // probability than the minimum node in the pool, we need to
              // remove the minimum node from the pool and add the new token to
              // the tree
              tokens.erase(tokens.begin());
            }
            tokens.insert(node_ptr);
          }
        }
      }
      parent_pos++;
    }

    // Now add all tokens in the set to the token tree, in descending order of
    // their joint probability
    spec_token_tree.add_layer();
    for (auto token_it = tokens.crbegin(); token_it != tokens.crend();
         token_it++) {
      token_pool_full =
          token_tree_node_pool.size() == get_max_tokens_per_batch();
      if (token_pool_full and
          token_tree_node_pool.top().first->log_accumulated_prob >=
              (*token_it)->log_accumulated_prob) {
        break;
      } else if (token_pool_full) {
        token_tree_node_pool.top().first->pruned = true;
        all_requests[token_tree_node_pool.top().second]
            .speculative_token_trees[0]
            .tree_size--;
        token_tree_node_pool.pop();
      }

      token_tree_node_pool.push(std::make_pair((*token_it), guid));
      spec_token_tree.tree_layers.back().push_back((*token_it));
      spec_token_tree.tree_size++;
    }
  }

  bool all_request_last_layer_empty = true;

  for (int request_index = 0; request_index < get_max_requests_per_batch();
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    TokenTree &spec_token_tree = request.speculative_token_trees[0];

    if (spec_token_tree.tree_layers.size() <= current_speculation_step) {
      // This request has no token added in this layer, skip it
      continue;
    }

    std::list<std::shared_ptr<TokenTreeNode>> &last_layer =
        spec_token_tree.tree_layers.back();
    for (auto it = last_layer.begin(); it != last_layer.end();) {
      if ((*it)->pruned) {
        it = last_layer.erase(it);
        // spec_token_tree.tree_size--;
      } else {
        ++it;
      }
    }
    all_request_last_layer_empty &= last_layer.empty();

    if (last_layer.empty()) {
      spec_token_tree.tree_layers.pop_back();
    }
  }
  assert(token_tree_node_pool.size() <= get_max_tokens_per_batch() &&
         "The token tree node pool should not exceed the maximum size.");
  return all_request_last_layer_empty;
}

std::ostream &operator<<(std::ostream &os, TokenTree const &token_tree) {
  os << "Token tree: " << std::endl;
  int layer_idx = 0;
  for (auto const &layer : token_tree.tree_layers) {
    os << "Layer: " << layer_idx << std::endl;
    int token_pos = 0;
    for (auto const &node : layer) {
      if (!node->pruned) {
        os << "token pos: " << token_pos << "token id: " << node->id << "\t"
           << "parent pos: " << node->parent_pos << "\t" << std::endl;
      }
      token_pos++;
    }
    layer_idx++;
  }
  return os;
}

/* --------- Request Token Tree Related Functions --------- */
}; // namespace FlexFlow
