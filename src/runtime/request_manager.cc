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
      total_request_run_time(0.0f), request_manager_status(PREFILLING) {
  // The following config parameters are set
  // during ffmodel.compile()
  // Initialize them to -1 to make sure no one
  // gets an incorrect value of them before
  // ffmodel.compile()
  max_requests_per_batch = -1;
  max_tokens_per_batch = -1;
  max_spec_tree_token_num = -1;
  max_sequence_length = -1;
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
  return max_tokens_per_batch +
         max_spec_tree_token_num * max_requests_per_batch;
}

void RequestManager::set_max_sequence_length(int max_seq_length) {
  assert(max_sequence_length == -1 || max_sequence_length == max_seq_length);
  max_sequence_length = max_seq_length;
}

int RequestManager::get_max_sequence_length() {
  assert(max_sequence_length > 0);
  return max_sequence_length;
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
    RequestManager::register_new_request(std::vector<TokenId> const &prompt,
                                         int max_sequence_length) {
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
    RequestManager::register_new_request(std::string const &prompt,
                                         int max_sequence_length) {
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
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (guid_of_requests[i] != INVALID_GUID) {
      count++;
    }
  }
  return count;
}

int RequestManager::get_empty_request_index() {
  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
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
  Request &new_request = pending_request_queue.front();
  all_requests[new_request.guid] = new_request;
  BatchConfig::RequestGuid guid = new_request.guid;
  pending_request_queue.pop();
  prefill_request = std::make_shared<Request>(all_requests[guid]);

  // Find an empty slot
  int request_index = get_empty_request_index();
  assert(request_index != -1 && "No empty request slot to load the request.");
  prefill_request->batch_index = request_index;
  guid_of_requests[request_index] = guid;
  request_available[request_index] = true;
  num_available_requests++;
  request_available[request_index] = true;
}

void RequestManager::update_inference_results(InferenceResult const &result) {
  // Update the inference results
  std::lock_guard<std::mutex> const lock(rm_state_mutex);
  switch (request_manager_status) {
    case PREFILLING:
      if (decoding_mode == INCREMENTAL_DECODING) {
        if (update_llm_prefill_results(result)) {
          // This indicates that the prefilling phase finishes
          request_manager_status = DECODING;
        }
      } else if (decoding_mode == SPECULATIVE_DECODING) {
        if (prefill_model == SSM) {
          if (update_ssm_prefill_results(result)) {
            // This indicates that the prefilling phase for SSM finishes
            // We need to start the LLM prefilling
            prefill_model = LLM;
          }
        } else if (prefill_model == LLM) {
          if (update_llm_prefill_results(result)) {
            // This indicates that the prefilling phase finishes
            request_manager_status = SSM_SPEC;
            // Reset the prefill_request
            prefill_request = nullptr;
          }
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
        } else {
          request_manager_status = PREFILLING;
          load_pending_reqeust_to_batch();
          prefill_model = SSM;
        }
      }
      break;
    case SSM_SPEC:
      SsmInferenceResult const &ssm_result =
          dynamic_cast<SsmInferenceResult const &>(result);
      if (update_ssm_inference_results(ssm_result)) {
        // Stop condition for the speculation phase has been reached
        request_manager_status = LLM_VERIFY;
      }
      // else, keep the current status
      break;
    default:
      assert(false && "Invalid request manager status.");
  }
}

// TO BE REMOVED: START
// void RequestManager::update_inference_results(InferenceResult const &result)
// {
//   // Update the inference results
//   std::lock_guard<std::mutex> const lock(rm_state_mutex);
//   for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
//     if (guid_of_requests[i] == INVALID_GUID) {
//       continue;
//     }
//     Request &request = all_requests[guid_of_requests[i]];

//     switch (request_manager_status) {
//       case PREFILLING:
//         if (request.initial_len ==
//             request.llm_cache_size) { // all prompt tokens are prefilled
//           request.tokens.push_back(
//               result.token_ids[request.num_tokens_in_batch]);
//           request_manager_status = DECODING;
//         }
//         break;
//       case DECODING:
//         request.tokens.push_back(
//             result.token_ids[request.first_token_offset_in_batch]);
//         if (request.tokens.size() ==
//             request.max_sequence_length) { // request is completed
//           request.status = Request::COMPLETED;
//           trigger_request_completion_future(request.guid);
//           guid_of_requests[i] = INVALID_GUID;
//           request_manager_status = PREFILLING;
//         }
//         break;
//       default:
//         assert(false);
//     }
//   }
// }
// TO BE REMOVED: END

bool RequestManager::update_llm_prefill_results(InferenceResult const &result) {
  // TODO:
  // The pending request can be found at Request_manager.prefill_request
  // 1. Update request.llm_cache_size
  // 2. Check if the prefilling is finished (request.tokens.size() ==
  // request.llm_cache_size)
  // 3. If the prefilling is finished, push the last token in result to
  // request.tokens
  // 4. Otherwise, no need to push
  // 5. Return true if the prefilling is finished
}

bool RequestManager::update_llm_decode_results(InferenceResult const &result) {
  // TODO:
  // 1. Iterate over all requests
  // request.num_tokens_in_batch != 0
  // 2. Check if the prefilling is finished
  // 3. If at least one request is completed, return true
}

bool RequestManager::update_ssm_prefill_results(
    InferenceResult const &ssm_prefill_result) {
  // This function is called by update_inference_results when the
  // request_manager_status is PREFILLING and the prefill_model is SSM.
  // There's no results to update, but we should update some SSM related states
  // related to SSM.
  prefill_request->ssm_cache_size += prefill_request->num_tokens_in_batch;
  if (prefill_request->ssm_cache_size == prefill_request->tokens.size()) {
    return true;
  }
  return false;
}

BatchConfig RequestManager::prepare_next_batch() {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);

  switch (request_manager_status) {
    case PREFILLING:
      return prepare_prefilling_batch();
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

BatchConfig RequestManager::prepare_prefilling_batch() {
  // This function is called when the request_manager_status is PREFILLING,
  // which means that there is a request in the prefilling phase.
  // This function load its prefilling tokens, constructing a BatchConfig with
  // only one request.

  BatchConfig bc;
  bc.prompt_phase = true;

  assert(prefill_request != nullptr &&
         "No prefilling request to process in the prefilling phase.");
  int request_index = prefill_request->batch_index;

  std::copy(std::begin(request_available),
            std::end(request_available),
            std::begin(bc.request_available));
  bc.request_available[request_index] = true;
  bc.num_available_requests = num_available_requests;

  bc.requestsInfo[request_index].first_token_offset_in_batch = 0;
  if (prefill_model == SSM) {
    // Request Info
    bc.requestsInfo[request_index].first_token_index_in_request =
        prefill_request->ssm_cache_size;
    bc.requestsInfo[request_index].num_tokens_in_batch = std::min(
        BatchConfig::MAX_NUM_TOKENS,
        (int)prefill_request->tokens.size() - prefill_request->ssm_cache_size);

  } else if (prefill_model == LLM) {
    // Request Info
    bc.requestsInfo[request_index].first_token_index_in_request =
        prefill_request->llm_cache_size;
    bc.requestsInfo[request_index].num_tokens_in_batch = std::min(
        BatchConfig::MAX_NUM_TOKENS,
        (int)prefill_request->tokens.size() - prefill_request->llm_cache_size);
  }

  prefill_request->first_token_offset_in_batch = 0;
  prefill_request->num_tokens_in_batch =
      bc.requestsInfo[request_index].num_tokens_in_batch;

  // Token Info
  for (int token_idx = 0;
       token_idx < bc.requestsInfo[request_index].num_tokens_in_batch;
       token_idx++) {
    int abs_idx = -1;
    if (prefill_model == SSM) {
      abs_idx = prefill_request->ssm_cache_size + token_idx;
    } else if (prefill_model == LLM) {
      abs_idx = prefill_request->llm_cache_size + token_idx;
    } else {
      assert(false && "Invalid prefill model.");
    }
    assert(abs_idx < prefill_request->tokens.size());

    bc.tokensInfo[token_idx].request_index = request_index;
    bc.tokensInfo[token_idx].abs_index_in_request = abs_idx;
    bc.tokensInfo[token_idx].token_id = prefill_request->tokens[abs_idx];

    bc.num_tokens++;
    prefill_request->num_tokens_in_batch++;
    // TODO: move the following line to update_inference_results
    // prefill_request->llm_cache_size++;
  }

  return bc;
}

BatchConfig RequestManager::prepare_decoding_batch() {
  // This function is called when the request_manager_status is DECODING. It
  // fills the last token of each request in the current batch to the
  // BatchConfig for the LLM to decode.

  BatchConfig bc;
  bc.prompt_phase = false;

  for (int i = 0; i < BatchConfig::MAX_NUM_REQUESTS; i++) {
    if (!request_available[i]) {
      continue;
    }
    bc.request_available[i] = true;
    bc.num_available_requests++;

    Request &request = all_requests[guid_of_requests[i]];

    // Per Request Info
    bc.requestsInfo[i].first_token_index_in_request = request.llm_cache_size;
    bc.requestsInfo[i].first_token_offset_in_batch = bc.num_tokens;
    bc.requestsInfo[i].num_tokens_in_batch = 1;

    request.first_token_offset_in_batch = bc.num_tokens;
    request.num_tokens_in_batch = 1;

    // Per Token Info
    bc.tokensInfo[bc.num_tokens].request_index = i;
    bc.tokensInfo[bc.num_tokens].abs_index_in_request = request.llm_cache_size;
    bc.tokensInfo[bc.num_tokens].token_id = request.tokens.back();

    // TODO: this should be updated in the update_inference_results() function
    request.llm_cache_size++;
    bc.num_tokens++;
  }

  return bc;
}
/* ----- Speculative Inference Specific functions ----- */

/***** Request Init Phase *****/
TreeSearchBatchConfig RequestManager::prepare_first_spec_batch_config() {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);
  if (verbose) {
    std::cout << "\n############### prepare_first_spec_batch_config "
                 "##############\n";
  }
  // TODO: Clean up the code, this method does the following:
  // 1. Commit the verified tokens through TreeSearchBatchConfig. We can do
  // this request by request. The infomation of the committed tokens are
  // stored in Request.ssm_committed_tokens. Put the information of the
  // committed tokens into BatchConfig.TokensInfo.
  // 2. Maintain BatchConfig::RequestsInfo and all other fields of
  // TreeSearchBatchConfig.
  // Please refer to the implementation of prepare_next_spec_batch_config()
  // for more details.
  TreeSearchBatchConfig new_bc;
  // Assume that only one small model is in use now
  new_bc.model_id = 0;
  new_bc.num_tokens = 0;
  new_bc.current_depth = 0;
  new_bc.num_available_requests = 0;
  new_bc.prompt_phase = true;
  assert(current_speculation_step == 0);

  for (int request_index = 0; request_index < BatchConfig::MAX_NUM_REQUESTS;
       ++request_index) {
    if (!request_available[request_index]) {
      new_bc.request_available[request_index] = false;
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    new_bc.request_available[request_index] = true;
    new_bc.num_available_requests++;
    // TODO: check this profiling, what is profiling
    profiling_requests[request.guid].ssm_decoding_steps += 1;

    std::vector<Request::CommittedToken> &committed_tokens =
        request.committed_tokens;

    // 2. Maintain requestsInfo
    new_bc.requestsInfo[request_index].first_token_offset_in_batch =
        new_bc.num_tokens;
    new_bc.requestsInfo[request_index].first_token_index_in_request =
        request.tokens.size() - committed_tokens.size();
    new_bc.requestsInfo[request_index].num_tokens_in_batch =
        committed_tokens.size();

    // 3. Store committed tokens to tokensInfo
    for (int committed_token_index = 0;
         committed_token_index < committed_tokens.size();
         committed_token_index++) {
      Request::CommittedToken &committed_token =
          committed_tokens.at(committed_token_index);
      new_bc.tokensInfo[new_bc.num_tokens].request_index = request_index;
      new_bc.tokensInfo[new_bc.num_tokens].abs_index_in_request =
          committed_token.to_index;
      new_bc.tokensInfo[new_bc.num_tokens].token_id = committed_token.token_id;
      new_bc.num_tokens++;
    }
    // 4. Copy the causal mask, it should already been updated
    new_bc.causalMask[request_index] = request.causal_mask;
  }
  if (verbose) {
    std::cout << "prepare_first_spec_batch_config NEW batchconfig:"
              << std::endl;
    new_bc.print();
  }
  return new_bc;
}

/***** Speculative Decoding Phase *****/
TreeSearchBatchConfig RequestManager::prepare_next_spec_batch_config() {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);
  if (verbose) {
    std::cout << "\n############### prepare_next_batch_spec ###############\n";
    std::cout << "Current tree depth: " << current_speculation_step << "\n";
  }
  // Prepare the next batch for existing requests
  TreeSearchBatchConfig new_bc;
  // We assume that only one small model is in use now
  new_bc.model_id = 0;
  new_bc.num_tokens = 0;
  new_bc.current_depth = current_speculation_step;
  new_bc.num_available_requests = 0;
  new_bc.prompt_phase = false;

  for (int request_index = 0; request_index < BatchConfig::MAX_NUM_REQUESTS;
       ++request_index) {
    if (!request_available[request_index]) {
      new_bc.request_available[request_index] = false;
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    new_bc.request_available[request_index] = true;
    new_bc.num_available_requests++;
    new_bc.requestsInfo[request_index].first_token_offset_in_batch =
        new_bc.num_tokens;
    // TODO: check this profiling
    profiling_requests[request.guid].ssm_decoding_steps += 1;

    // Fill in the tokens
    TokenTree &token_tree = request.speculative_token_trees.at(new_bc.model_id);
    if (token_tree.tree_layers.size() <= current_speculation_step) {
      // This request has no token to decode in this and the following small
      // model inference steps
      new_bc.requestsInfo[request_index].num_tokens_in_batch = 0;
      new_bc.requestsInfo[request_index].first_token_index_in_request =
          request.tokens.size() + token_tree.tree_size_including_pruned;
      continue;
    } else {
      std::list<std::shared_ptr<TokenTreeNode>> &current_layer =
          token_tree.tree_layers.at(current_speculation_step);
      // Exclude the current layer from the token tree, because we want the
      // start index
      new_bc.requestsInfo[request_index].first_token_index_in_request =
          request.tokens.size() + token_tree.tree_size_including_pruned -
          current_layer.size();
      new_bc.requestsInfo[request_index].num_tokens_in_batch =
          current_layer.size();

      int child_index = 0;
      for (auto const &node_ptr : current_layer) {
        new_bc.tokensInfo[new_bc.num_tokens].request_index = request_index;
        new_bc.tokensInfo[new_bc.num_tokens].abs_index_in_request =
            new_bc.requestsInfo[request_index].first_token_index_in_request +
            child_index;
        new_bc.tokensInfo[new_bc.num_tokens].token_id = node_ptr->id;

        new_bc.num_tokens++;
        child_index++;
      }
    }

    // Copy the causal mask, it should already been updated
    new_bc.causalMask[request_index] = request.causal_mask;
  }

  if (verbose) {
    std::cout << "prepare_next_batch_beam NEW batchconfig:" << std::endl;
    new_bc.print();
  }
  return new_bc;
}

/***** Verify Phase *****/
TreeVerifyBatchConfig RequestManager::prepare_verify_batch_config() {
  std::lock_guard<std::mutex> const lock(request_queue_mutex);
  if (verbose) {
    std::cout
        << "\n############### prepare_next_batch_verify ###############\n";
  }
  // TODO: Clean up the code, this method does the following:
  // 1. Commit the verified tokens in the last iteration through the
  // TreeVerifyBatchConfig. We can do this request by request.
  // The information of the committed tokens is stored in
  // Request.llm_committed_tokens. Put the information of the committed tokens
  // into TreeVerifyBatchConfig.committed_tokens.
  // 2. Load the tokens on the token tree that are not yet pruned to
  // TreeVerifyBatchConfig.tokensInfo. Be careful with the abs_depth etc.
  // (skip the pruned tokens).
  // 3. Create the causal mask for the large model based on the small model
  // causal mask (call create_llm_bitmask()).
  // 4. Maintain TreeVerifyBatchConfig::RequestsInfo and all other fields of
  // TreeSearchBatchConfig.
  // Please refer to the implementation of prepare_next_spec_batch_config()
  // for more details.
  TreeVerifyBatchConfig new_bc;
  new_bc.num_tokens = 0;
  new_bc.num_available_requests = 0;
  new_bc.num_tokens_to_commit = 0;
  new_bc.prompt_phase = false;

  for (int request_index = 0; request_index < BatchConfig::MAX_NUM_REQUESTS;
       ++request_index) {
    if (!request_available[request_index]) {
      new_bc.request_available[request_index] = false;
      continue;
    }
    int guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    new_bc.request_available[request_index] = true;
    new_bc.num_available_requests++;
    // TODO: check this profiling
    profiling_requests[request.guid].llm_decoding_steps += 1;

    // 1. Maintain requestsInfo
    new_bc.requestsInfo[request_index].first_token_index_in_request =
        request.tokens.size();
    new_bc.requestsInfo[request_index].first_token_offset_in_batch =
        new_bc.num_tokens;

    // 2. Put the information of the committed tokens into
    // TreeVerifyBatchConfig.committed_tokens.
    // Note here, we shouldn't put the last token in request.committed_tokens
    // into new_bc. Because the LLM don't have that token's KV cache.
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

    // 3. Load the tokens on the token tree that are not yet pruned to
    // TreeVerifyBatchConfig.tokensInfo.
    TokenTree &token_tree = request.speculative_token_trees[0];
    int token_tree_index = 0;
    for (auto const &tree_layer : token_tree.tree_layers) {
      for (auto const &tree_node : tree_layer) {
        if (tree_node->pruned == false) {
          new_bc.tokensInfo[new_bc.num_tokens].request_index = request_index;
          new_bc.tokensInfo[new_bc.num_tokens].abs_index_in_request =
              request.tokens.size() + token_tree_index;
          new_bc.tokensInfo[new_bc.num_tokens].token_id = tree_node->id;
          new_bc.num_tokens++;
          token_tree_index++;
        }
      }
    }

    // 4. Maintain requestsInfo.num_tokens_in_batch of TreeSearchBatchConfig
    new_bc.requestsInfo[request_index].num_tokens_in_batch =
        token_tree_index + 1;

    // 5. Create the causal mask for the large model based on the small model
    // causal mask.
    new_bc.causalMask[request_index] = create_llm_bitmask(guid);
  }

  if (verbose) {
    std::cout << "prepare_next_batch_verify NEW batchconfig:" << std::endl;
    new_bc.print();
  }
  return new_bc;
}

bool RequestManager::update_llm_verify_results(
    InferenceResult const &llm_verify_result) {
  // TODO: Implement this function
  // We may have two types of InferenceResults, one is the results from
  // sampling the large model, the other is the top-p / top-k logits of the
  // large model, we can first implement the former one. For the latter one,
  // we have to add a CPU based verify function.
  // 1. Compare the results returned from the LLM and compare them with the
  // SSM's speculative token tree. For the greedy construction of the
  // speculative token tree, we can simply compare LLM's sample result at each
  // token, this is implemented in get_verify_results_greedy(). This function
  // stores the commmitted tokens into the corresponding fields in the
  // Request. For the sampling construction of the speculative token tree, we
  // need to implement a CPU based verify function.
  // 2. Call init_token_tree() add_root_token_to_spec_token_tree() to add the
  // root token to the requests' speculative token tree. The root token is the
  // last committed token.
  // 3. For requests not completed, update their causal mask.
  // 4. Some requests may be completed after appending the verified tokens. If
  // there is a request completed, return true.
  get_verify_results_greedy(llm_verify_result);
}

bool RequestManager::update_ssm_inference_results(
    SsmInferenceResult const &ssm_inference_result) {
  // This function returns false if no tokens are added to the token tree,
  // which indicates that the ssm inference phase is done.
  assert(current_speculation_step >= 1 &&
         "The current speculation step should be no less than 1");

  int num_branches = TreeSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
  int result_index = 0;
  bool token_added_to_spec_tree = false;

  // Here we assume that the order of the tokens in the last
  // TreeSearchBatchConfig and hence the last SsmInferenceResult is equal to
  // the order of the request in the last TreeSearchBatchConfig
  for (int request_index = 0; request_index < BatchConfig::MAX_NUM_REQUESTS;
       ++request_index) {
    if (!request_available[request_index]) {
      // Request in this slot is unavailable
      continue;
    }
    FlexFlow::RequestManager::RequestGuid guid =
        guid_of_requests[request_index];
    Request &request = all_requests[guid];

    TokenTree &token_tree = request.speculative_token_trees[0];
    if (token_tree.tree_layers.size() < current_speculation_step) {
      // This means that the parent layer is empty
      continue;
    } else {
      std::list<std::shared_ptr<TokenTreeNode>> &parent_tree_layer =
          token_tree.tree_layers[current_speculation_step - 1];
      int parent_pos = 0;
      for (auto parent_it = parent_tree_layer.begin();
           parent_it != parent_tree_layer.end();
           parent_it++) {
        if ((*parent_it)->pruned) {
          // Parent token is pruned, we have to skip all its children
          // Because no token is pruned in the last layer during the small
          // model inference, the reason why some parents are pruned is that
          // adding tokens to the new layer of the tree may result in some
          // node being pruned in internal layers.
          result_index += num_branches;
        } else {
          // Parent token is not pruned
          for (int child_idx = 0; child_idx < num_branches; child_idx++) {
            float parent_prob = (*parent_it)->joint_prob;
            token_added_to_spec_tree =
                token_added_to_spec_tree ||
                add_token_to_spec_token_tree(
                    guid,
                    ssm_inference_result.token_ids[result_index],
                    ssm_inference_result.probs[result_index] * parent_prob,
                    parent_pos);
            result_index++;
          }
        }
        parent_pos++;
      }
    }
    append_bitmask(guid);
  }
  return token_added_to_spec_tree;

  /* Move this to update_inference_results() */
  // State maintenance
  current_speculation_step++;
  if (!token_added_to_spec_tree ||
      current_speculation_step > TreeSearchBatchConfig::MAX_TREE_DEPTH) {
    // No token is added to the token tree, which indicates that the ssm
    // inference phase is done. Proceed to the large model verification phase.
    request_manager_status = LLM_VERIFY;
  }
}

/* --------- Bitmask Related Functions --------- */

void RequestManager::init_bitmask_prompt(RequestGuid guid, int prompt_length) {
  // This method is called by update_llm_verify_results when there are new
  // request to load into the batch
  // 1. Clear the causal mask because our current speculative token tree is
  // empty.
  // 2. Maintain all other fields.
  Request &request = all_requests[guid];
  BatchConfig::BitMask &bitmask = request.causal_mask;

  bitmask.clear_bitmask();
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
  bitmask.clear_bitmask();
  bitmask.tree_or_prompt_size = num_committed_tokens;
  bitmask.current_layer_size = num_committed_tokens;
}

void RequestManager::init_bitmask_spec(RequestGuid guid,
                                       int num_committed_tokens) {
  // This method modifies the bitmask in place
  // This method is called by the first call of update_ssm_verify_results in a
  // speculative iteration
  // CAUTION: You should still call append_bitmask() after this method
  // 1. Clear the causal mask and add a root into it, because the tree is
  // currently empty but we have a root.
  // 2. Maintain all other fields.
  assert(current_speculation_step == 1 &&
         "The current speculation step should be 1");
  Request &request = all_requests[guid];
  BatchConfig::BitMask &bitmask = request.causal_mask;
  bitmask.clear_bitmask();
  // Set the mask for the root
  bitmask.bit_mask[0].set_bit(0);
  bitmask.tree_or_prompt_size = 1;
  bitmask.non_tree_cache_size += num_committed_tokens;
  bitmask.current_layer_size = 1;
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
      request.speculative_token_trees[0].tree_layers[current_speculation_step];
  int new_layer_size = tree_layer.size();
  int last_layer_size = bitmask.current_layer_size;
  int previous_tree_size = bitmask.tree_or_prompt_size;
  bitmask.current_layer_size = new_layer_size;
  bitmask.tree_or_prompt_size += new_layer_size;

  assert(bitmask.tree_or_prompt_size <= BatchConfig::MAX_SPEC_TREE_TOKEN_NUM);

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
  // prepare_verify_batch_config()
  // TODO: implement this function
  // 1. Create the bitmask based on the pruned request token tree
  // 2. Maintain all other fields

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
  int llm_result_offset = 0;
  // This function maintain the generated token list of the request and the
  // committed tokens.
  for (int request_index = 0; request_index < BatchConfig::MAX_NUM_REQUESTS;
       ++request_index) {
    if (!request_available[request_index]) {
      continue;
    }
    RequestGuid guid = guid_of_requests[request_index];
    Request &request = all_requests[guid];
    assert(request.status == Request::RUNNING);
    request.committed_tokens.clear();

    int committed_token_index = request.tokens.size();

    TokenTree &token_tree = request.speculative_token_trees[0];
    // First add the root to the committed tokens
    request.committed_tokens.push_back(Request::CommittedToken(
        llm_result_offset,
        committed_token_index,
        llm_verify_result.token_ids[llm_result_offset]));
    committed_token_index++;
    // The position of the last accepted token in its tree layer
    int last_accepted_token_layer_index = 0;
    // The index of the last accepted token in the entire tree (excluding the
    // pruned tokens)
    int last_accepted_token_index = 0;

    int current_token_index = 1; // Because we skip the root
    int num_layers = token_tree.tree_layers.size();
    for (int layer_index = 1; layer_index < num_layers; layer_index++) {
      // We skip the first layer
      std::list<std::shared_ptr<TokenTreeNode>> &tree_layer =
          token_tree.tree_layers.at(layer_index);

      bool token_accepted_this_layer = false;
      int current_token_layer_index = 0;

      for (auto const &node_ptr : tree_layer) {
        if (node_ptr->pruned) {
          continue;
        }
        if ((node_ptr->parent_pos != last_accepted_token_layer_index) ||
            token_accepted_this_layer) {
          // The token's parent is not accepted, or there is already another
          // token accepted in this layer
          current_token_index++;
          current_token_layer_index++;
          continue;
        } else {
          // The token's parent is accepted, and no token has been accepted in
          // this layer yet
          if (node_ptr->id ==
              llm_verify_result
                  .token_ids[llm_result_offset + last_accepted_token_index]) {
            // The token's parent is accepted, and this token's id equals the
            // llm's sample at its parent's position. We accept this token.

            // from_index: the index of the token in the tree
            // to_index: the committed token index in the request
            request.committed_tokens.push_back(Request::CommittedToken(
                current_token_index, committed_token_index, node_ptr->id));
            request.tokens.push_back(node_ptr->id);

            token_accepted_this_layer = true;
            last_accepted_token_index = current_token_index;
            last_accepted_token_layer_index = current_token_layer_index;
            committed_token_index++;
            current_token_index++;
            current_token_layer_index++;
          }
        }
      }
      if (!token_accepted_this_layer) {
        // No token is accepted in this layer, we should stop the traversal
        // However, we have to add the last sampled token as a correction from
        // the LLM

        // from_index: since this token is not in the token tree, neither the
        // ssm nor the llm have its KV cache, so the from_index should be a
        // place holder, which is -1
        request.committed_tokens.push_back(Request::CommittedToken(
            -1,
            committed_token_index,
            llm_verify_result
                .token_ids[llm_result_offset + last_accepted_token_index]));
        request.tokens.push_back(
            llm_verify_result
                .token_ids[llm_result_offset + last_accepted_token_index]);
        break;
      }
    }
    llm_result_offset += token_tree.tree_size;
  }
}

std::vector<GenerationResult>
    FFModel::generate(std::vector<std::string> &prompts, int max_seq_length) {
  RequestManager *rm = RequestManager::get_request_manager();
  std::vector<RequestManager::RequestGuid> guids;
  for (int i = 0; i < prompts.size(); i++) {
    RequestManager::RequestGuid guid =
        rm->register_new_request(prompts.at(i), max_seq_length);
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
  if (rm->get_num_ssms() == 0) {
    // No SSMs: perform incremental decoding
    // rm->serve_incr_decoding(llm);
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

// TO BE REMOVED: START
// void RequestManager::serve_incr_decoding(FFModel *llm) {
//   Context ctx = llm->config.lg_ctx;
//   Runtime *runtime = llm->config.lg_hlr;
//   // Compile the llm
//   InferenceManager *im = InferenceManager::get_inference_manager();
//   im->compile_model_and_allocate_buffer(llm);
//   assert(im->model_weights_loaders.find(llm) !=
//          im->model_weights_loaders.end());
//   // Load model weights
//   im->model_weights_loaders[llm]->load_weights(llm);
//   // init operators
//   im->init_operators_inference(llm);
//   // Legion futures for inc_decoding and spec_infer
//   BatchConfigFuture last_bcf;
//   InferenceResultFuture last_irf;
//   {
//     // Initialize futures for incr decoding
//     BatchConfig bc;
//     InferenceResult ir;
//     last_bcf = Future::from_value<BatchConfig>(bc);
//     last_irf = Future::from_value<InferenceResult>(ir);
//   }

//   std::queue<std::pair<BatchConfigFuture, InferenceResultFuture>>
//       batch_pipeline;
//   { batch_pipeline.push(std::make_pair(last_bcf, last_irf)); }

//   while (!is_background_server_terminated()) {

//     if (batch_pipeline.size() >= 4) {
//       // Block here to avoid launching too many batches
//       auto const &batch = batch_pipeline.front();
//       batch.second.get_void_result();
//     }
//     // deque finished batches
//     while (batch_pipeline.size() > 1) {
//       auto const &batch = batch_pipeline.front();
//       if (batch.second.is_ready()) {
//         batch_pipeline.pop();
//       } else {
//         break;
//       }
//     }
//     runtime->begin_trace(ctx, 12346 /*trace_id*/);
//     auto const &next_batch = batch_pipeline.back();
//     BatchConfigFuture bcf =
//         prepare_next_batch(next_batch.first, next_batch.second, ctx,
//         runtime);
//     FutureMap fm = im->inference(llm, 0, bcf);
//     assert(fm.get_future_map_domain().get_volume() == 1);
//     InferenceResultFuture irf = fm.get_future(0);
//     batch_pipeline.push(std::make_pair(bcf, irf));
//     last_bcf = bcf;
//     last_irf = irf;
//     runtime->end_trace(ctx, 12346 /*trace_id*/);
//   }
// }
// TO BE REMOVED: END

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
    if (request_manager_status == PREFILLING) {
      if (prefill_model == LLM) {
        FutureMap fm = im->inference(llm, 0, bcf);
        assert(fm.get_future_map_domain().get_volume() == 1);
        InferenceResultFuture irf = fm.get_future(0);
        batch_pipeline.push(irf);
      } else if (prefill_model == SSM) {
        FutureMap fm = im->inference(get_ssm_model(0), 0, bcf);
        assert(fm.get_future_map_domain().get_volume() == 1);
        InferenceResultFuture irf = fm.get_future(0);
        batch_pipeline.push(irf);
      } else {
        assert(false && "Invalid prefill model");
      }
    } else if (request_manager_status == LLM_VERIFY) {
      FutureMap fm = im->inference(llm, 0, bcf);
      assert(fm.get_future_map_domain().get_volume() == 1);
      InferenceResultFuture irf = fm.get_future(0);
      batch_pipeline.push(irf);
    } else if (request_manager_status == SSM_SPEC) {
      FutureMap fm = im->inference(get_ssm_model(0), 0, bcf);
      assert(fm.get_future_map_domain().get_volume() == 1);
      InferenceResultFuture irf = fm.get_future(0);
      batch_pipeline.push(irf);
    } else {
      assert(false && "Invalid request manager status");
    }
    runtime->end_trace(ctx, 12345 /*trace_id*/);
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
  // tree, with parent_pos being -1 and joint_prob being 1.0
  Request &request = all_requests[guid];
  TokenTree &speculative_token_tree = request.speculative_token_trees[0];
  speculative_token_tree.add_layer();
  auto node_ptr = std::make_shared<TokenTreeNode>(token_id, -1, 1.0);
  speculative_token_tree.tree_layers[0].push_back(node_ptr);
  speculative_token_tree.tree_size++;
  speculative_token_tree.tree_size_including_pruned++;
}

bool RequestManager::add_token_to_spec_token_tree(RequestGuid guid,
                                                  BatchConfig::TokenId token_id,
                                                  int parent_pos,
                                                  float joint_prob) {
  // This method assumes only one small model is used for speculation
  // This method is called by update_ssm_inference_results()

  // This is called after the first small model inference
  assert(current_speculation_step >= 1 &&
         "The current speculation step should be no less than 1");

  // First make sure there are enough layers in the speculation tree
  Request &request = all_requests[guid];
  TokenTree &speculative_token_tree = request.speculative_token_trees[0];

  if (speculative_token_tree.tree_layers.size() == current_speculation_step) {
    // When adding the first token, we need to add a new layer
    speculative_token_tree.add_layer();
  } else {
    // To add a token, the tree depth is either the same as the current
    // speculation step or one more than the current speculation step.
    assert(speculative_token_tree.tree_layers.size() ==
               current_speculation_step + 1 &&
           "Invalid token tree depth");
  }

  bool remove_min_node = false;
  bool add_new_node = true;

  std::shared_ptr<TokenTreeNode> min_node_ptr = nullptr;
  RequestGuid min_node_guid = -1;
  if (token_tree_node_pool.size() > 0) {
    std::pair<std::shared_ptr<TokenTreeNode>, RequestGuid>
        min_node_pair_in_pool = token_tree_node_pool.top();
    min_node_ptr = min_node_pair_in_pool.first;
    min_node_guid = min_node_pair_in_pool.second;
  }

  // We maintain the size of the token tree node pool to not exceed
  //  BatchConfig::MAX_NUM_TOKENS
  if (token_tree_node_pool.size() == BatchConfig::MAX_NUM_TOKENS) {
    // The pool is full, check if the new node has a higher joint probability
    // than the minimum node in the pool.

    if (joint_prob < min_node_ptr->joint_prob) {
      // Insertion failed
      add_new_node = false;
    } else {
      // Remove the minimum node from the pool, and set its pruned field to
      // true
      remove_min_node = true;
    }
  } else if (token_tree_node_pool.size() > BatchConfig::MAX_NUM_TOKENS) {
    assert(false && "The size of the token tree node pool should not exceed "
                    "BatchConfig::MAX_NUM_TOKENS");
  }
  // Do nothing if the pool is not full

  // The request's token tree size should not exceed
  // BatchConfig::MAX_SPEC_TREE_TOKEN_NUM
  // The judgement is done here to avoid the case where the tree is full but a
  // node is pruned.
  if (speculative_token_tree.tree_size ==
      BatchConfig::MAX_SPEC_TREE_TOKEN_NUM) {
    if (remove_min_node && guid == min_node_guid) {
      // The minimum node in the pool is pruned, and it's in the same request
      // with the new node. Only in this case we can add the new node.
      // Because remove_min_node is true means that the new node has a higher
      // joint probability than the minimum node in the pool.
      add_new_node = true;
    } else {
      // Otherwise, we cannot add the new node, and we don't need to expel the
      // minimum node from the pool.
      add_new_node = false;
      remove_min_node = false;
    }
  } else if (speculative_token_tree.tree_size >
             BatchConfig::MAX_SPEC_TREE_TOKEN_NUM) {
    assert(false && "The size of the token tree should not exceed "
                    "BatchConfig::MAX_SPEC_TREE_TOKEN_NUM");
  }

  assert(!(remove_min_node && !add_new_node) &&
         "The minimum node should be removed only when the new node is added");

  if (remove_min_node) {
    // Remove the minimum node from the pool, and set its pruned field to true
    min_node_ptr->pruned = true;
    token_tree_node_pool.pop();
    all_requests[min_node_guid].speculative_token_trees[0].tree_size--;
  }

  if (add_new_node) {
    // Add the new node to the pool and the last layer of the speculation tree
    auto node_ptr =
        std::make_shared<TokenTreeNode>(token_id, parent_pos, joint_prob);
    token_tree_node_pool.push(std::make_pair(node_ptr, guid));
    request.speculative_token_trees[0]
        .tree_layers[current_speculation_step]
        .push_back(node_ptr);
    speculative_token_tree.tree_size++;
    speculative_token_tree.tree_size_including_pruned++;
  }
  return add_new_node;
}

void RequestManager::prune_last_layer_of_spec_token_tree(RequestGuid guid) {
  // This method assumes only one small model is used for speculation
  Request &request = all_requests[guid];

  if (request.speculative_token_trees[0].tree_layers.size() <=
      current_speculation_step) {
    // There are no tokens in the last layer
    return;
  }
  auto &last_layer =
      request.speculative_token_trees[0].tree_layers[current_speculation_step];
  for (auto it = last_layer.begin(); it != last_layer.end(); ++it) {
    if ((*it)->pruned) {
      last_layer.erase(it);
      request.speculative_token_trees[0].tree_size--;
    }
  }
}
/* --------- Request Token Tree Related Functions --------- */
}; // namespace FlexFlow
