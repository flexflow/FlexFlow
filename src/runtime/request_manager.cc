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
  assert(!fs.fail() && "no such file");
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

RequestManager::RequestManager()
    : request_manager_status(INITIALIZED), verbose(false),
      next_available_guid(1000000), num_processed_requests(0),
      total_request_run_time(0.0f) {
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

void RequestManager::push_spec_infer_tree_width(int tree_width) {
  assert(tree_width <= BeamSearchBatchConfig::MAX_BEAM_WIDTH);
  spec_infer_tree_width.emplace_back(tree_width);
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
  assert(model_id < ssm_models.size());
  return ssm_models[model_id];
}

size_t RequestManager::get_num_ssms() {
  return ssm_models.size();
}

RequestManager::RequestGuid
    RequestManager::register_new_request(std::vector<TokenId> const &prompt,
                                         int max_sequence_length) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);

  // Add a new request
  Request request;
  request.status = Request::PENDING;
  request.guid = next_available_guid++;
  request.max_sequence_length = max_sequence_length;

  if (prompt.size() >= get_max_sequence_length()) {
    std::cout << "Warning: too many tokens in prompt, only load up to "
              << get_max_sequence_length() << " tokens, but got "
              << prompt.size() << ".\n";

    printf("tokens size: %zu\n", request.tokens.size());
    return INVALID_GUID;
  } else {
    request.initial_len = prompt.size();
    request.tokens = prompt;
  }

  if (get_num_ssms() == 0) {
    std::cout << "No small speculative model registered, using incremental "
                 "decoding."
              << std::endl;
  } else {
    std::cout << "Num of SSMs: " << get_num_ssms() << std::endl;
    for (int i = 0; i < get_num_ssms(); i++) {
      BeamTree beam_tree = BeamTree{};
      request.beam_trees.push_back(beam_tree);
    }
  }

  pending_request_queue.push(request);
  all_requests[request.guid] = request;
  {
    const std::lock_guard<std::mutex> lock(request_to_promise_mutex);
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
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  // Add a new request
  Request request;
  request.status = Request::PENDING;
  request.guid = next_available_guid++;
  request.max_sequence_length = max_sequence_length;
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
  request.initial_len = request.tokens.size();

  if (get_num_ssms() == 0) {
    std::cout << "No small speculative model registered, using incremental "
                 "decoding."
              << std::endl;
  } else {
    std::cout << "Num of SSMs: " << get_num_ssms() << std::endl;
    for (int i = 0; i < get_num_ssms(); i++) {
      BeamTree beam_tree = BeamTree{};
      request.beam_trees.push_back(beam_tree);
    }
  }

  pending_request_queue.push(request);
  all_requests[request.guid] = request;
  {
    const std::lock_guard<std::mutex> lock(request_to_promise_mutex);
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
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
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
    const std::lock_guard<std::mutex> lock(request_to_promise_mutex);
    assert(request_to_promise.find(guid) != request_to_promise.end());
    future = request_to_promise[guid]->get_future();
  }
  // Wait until the result is completed
  future.get();
  // Get the generation result
  {
    const std::lock_guard<std::mutex> lock(request_queue_mutex);
    assert(request_generation_results.find(guid) !=
           request_generation_results.end());
    return request_generation_results[guid];
  }
}

size_t RequestManager::get_num_processed_requests() {
  return num_processed_requests;
}

BatchConfigFuture
    RequestManager::prepare_next_batch(BatchConfigFuture const &old_bc,
                                       InferenceResultFuture const &result,
                                       Context ctx,
                                       Runtime *runtime) {
  RequestManager *rm = this;
  TaskLauncher launcher(RM_PREPARE_NEXT_BATCH_TASK_ID,
                        TaskArgument(&rm, sizeof(RequestManager *)));
  launcher.add_future(old_bc);
  launcher.add_future(result);
  return runtime->execute_task(ctx, launcher);
}

BatchConfig RequestManager::prepare_next_batch_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  RequestManager *rm = *((RequestManager **)task->args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  InferenceResult const &result =
      Future(task->futures[1]).get_result<InferenceResult>();
  return rm->prepare_next_batch(*bc, result);
}

BatchConfig RequestManager::prepare_next_batch(BatchConfig const &old_bc,
                                               InferenceResult const &result) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);

  // Step 1: append result from previous iteration to request's tokens
  for (int i = 0; i < old_bc.num_tokens; i++) {
    size_t guid =
        old_bc.requestsInfo[old_bc.tokensInfo[i].request_index].request_guid;
    Request &request = all_requests[guid];
    if (old_bc.tokensInfo[i].abs_depth_in_request + 1 < request.tokens.size()) {
      // This is a prompt token
      continue;
    } else {
      assert(old_bc.tokensInfo[i].abs_depth_in_request + 1 ==
             request.tokens.size());
      // This is a decoding token
      log_req_mgr.print("Output token is: %d", result.token_ids[i]);
      request.tokens.push_back(result.token_ids[i]);
      // std::string output = this->tokenizer_->Decode(request.tokens);
      // log_req_mgr.print("Output: %s", output.c_str());
    }
  }
  int num_generation_tokens = 0;
  int num_active_req = -1;

  // Step 2: prepare the next batch for existing requests
  BatchConfig new_bc;
  for (int i = 0; i < BatchConfig::max_requests_per_batch(); i++) {
    if (old_bc.request_completed[i]) { // add new requests to the next batch
      continue;
    } else {
      assert(old_bc.requestsInfo[i].num_tokens_in_batch > 0);
      Request &request = all_requests[old_bc.requestsInfo[i].request_guid];
      int processed_tokens =
          old_bc.requestsInfo[i].first_token_depth_in_request +
          old_bc.requestsInfo[i].num_tokens_in_batch;
      assert(processed_tokens < request.tokens.size());
      bool request_completed = false;
      // printf("model_type = %d\n", this->model_type);
      if (request.tokens.size() >= old_bc.requestsInfo[i].max_sequence_length) {
        request_completed = true;
      } else if (request.tokens.back() == eos_token_id) {
        // Encounter EOS token id
        request_completed = true;
      }
      if (request_completed) {
        std::string output = this->tokenizer_->Decode(request.tokens);
        // Unlike Huggingface, the sentencepiece C++ library automatically
        // removes the BOS token
        if (model_type == ModelType::LLAMA &&
            request.tokens.at(0) == bos_token_id) {
          output = "<s> " + output;
        }
        {
          // update generation result
          GenerationResult &gr = request_generation_results[request.guid];
          assert(gr.guid == request.guid);
          gr.output_tokens = request.tokens;
          gr.output_text = output;
        }
        request.status = Request::COMPLETED;
        trigger_request_completion_future(request.guid);
        log_req_mgr.print("[Done] guid(%zu) final_length(%zu)",
                          old_bc.requestsInfo[i].request_guid,
                          request.tokens.size());
        log_req_mgr.print("Final output: %s", output.c_str());
        num_processed_requests++;
        ProfileInfo profile_info = profiling_requests[request.guid];
        profile_info.finish_time = Realm::Clock::current_time_in_microseconds();
        total_request_run_time +=
            profile_info.finish_time - profile_info.start_time;
        profiling_requests[request.guid] = profile_info;
        log_req_mgr.print(
            "[Profile] guid(%zu) llm_decoding_steps(%d) start(%.1lf) "
            "finish(%.1lf) latency(%.1lf)",
            request.guid,
            profile_info.llm_decoding_steps,
            profile_info.start_time,
            profile_info.finish_time,
            profile_info.finish_time - profile_info.start_time);
        // Write output to file if needed:
        if (!output_filepath.empty()) {
          std::ofstream outputFile(output_filepath, std::ios::app);
          if (outputFile.is_open()) {
            outputFile << "end-to-end latency: " << std::fixed
                       << std::setprecision(3) << total_request_run_time
                       << std::endl;
            outputFile << "num decoding steps: "
                       << profile_info.llm_decoding_steps << std::endl;
            outputFile << "token IDs: ";
            for (int i = 0; i < request.tokens.size(); i++) {
              outputFile << request.tokens[i];
              if (i < request.tokens.size() - 1) {
                outputFile << ",";
              }
            }
            outputFile << std::endl;
            outputFile << output;
            outputFile.close();
          } else {
            std::cout << "Unable to open the output file: " << output_filepath
                      << std::endl;
            assert(false);
          }
        }

      } else {
        new_bc.request_completed[i] = false;
        new_bc.requestsInfo[i].first_token_depth_in_request = processed_tokens;
        new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
        new_bc.requestsInfo[i].request_guid =
            old_bc.requestsInfo[i].request_guid;
        new_bc.requestsInfo[i].max_sequence_length =
            old_bc.requestsInfo[i].max_sequence_length;
        num_active_req++;
        new_bc.requestsInfo[num_active_req].batch_config_request_id = i;
        if (new_bc.requestsInfo[i].first_token_depth_in_request + 1 ==
            request.tokens.size()) {
          // Incremental phase
          new_bc.requestsInfo[i].num_tokens_in_batch = 1;
          num_generation_tokens++;
          new_bc.requestsInfo[i].prompt_phase = false;
        } else {
          // Prompt phase
          new_bc.requestsInfo[i].num_tokens_in_batch =
              std::min(get_max_tokens_per_batch() - new_bc.num_tokens,
                       (int)request.tokens.size() -
                           new_bc.requestsInfo[i].first_token_depth_in_request);
          new_bc.requestsInfo[i].prompt_phase = true;
        }
        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          int depth = new_bc.requestsInfo[i].first_token_depth_in_request + j;
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;
          assert(depth < request.tokens.size());
          new_bc.tokensInfo[new_bc.num_tokens].token_id = request.tokens[depth];
          new_bc.num_tokens++;
        }
        // Update profiling
        profiling_requests[new_bc.requestsInfo[i].request_guid]
            .llm_decoding_steps++;
      }
    }
  }
  new_bc.num_generation_tokens = num_generation_tokens;

  // Step 3: add new requests to the next batch
  for (int i = 0; i < BatchConfig::max_requests_per_batch(); i++) {
    if (new_bc.request_completed[i]) {
      if (!pending_request_queue.empty() &&
          new_bc.num_tokens < get_max_tokens_per_batch()) {
        Request new_request = pending_request_queue.front();
        pending_request_queue.pop();
        // all_requests[new_request.guid] = new_request;

        new_bc.requestsInfo[i].first_token_depth_in_request = 0;
        new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
        new_bc.requestsInfo[i].request_guid = new_request.guid;
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(get_max_tokens_per_batch() - new_bc.num_tokens,
                     (int)new_request.tokens.size());
        new_bc.requestsInfo[i].max_sequence_length =
            new_request.max_sequence_length;
        new_bc.request_completed[i] = false;
        new_bc.requestsInfo[i].prompt_phase = true;
        num_active_req++;
        new_bc.requestsInfo[num_active_req].batch_config_request_id = i;
        // add profile_info for the new request
        ProfileInfo profile_info;
        profile_info.llm_decoding_steps = 1;
        profile_info.start_time = Realm::Clock::current_time_in_microseconds();
        profiling_requests[new_request.guid] = profile_info;
        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          int depth = new_bc.requestsInfo[i].first_token_depth_in_request + j;
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;
          assert(depth < new_request.tokens.size());
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
              new_request.tokens[depth];
          new_bc.num_tokens++;
        }
        if (new_bc.num_tokens == get_max_tokens_per_batch()) {
          break;
        }
      }
    }
  }

  return new_bc;
}

/* ----- Speculative Inference Specific functions ----- */

/***** Request Init Phase *****/
BeamSearchBatchConfigFuture RequestManager::prepare_next_batch_init(
    TreeVerifyBatchConfigFuture const &old_bc,
    InferenceResultFuture const &result,
    int model_id,
    Context ctx,
    Runtime *runtime) {

  RequestManager *rm = this;
  TaskLauncher launcher(RM_PREPARE_NEXT_BATCH_INIT_TASK_ID,
                        TaskArgument(&rm, sizeof(RequestManager *)));
  launcher.add_future(old_bc);
  launcher.add_future(result);
  launcher.add_future(Future::from_value<int>(model_id));
  return runtime->execute_task(ctx, launcher);
}

BeamSearchBatchConfig RequestManager::prepare_next_batch_init_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  RequestManager *rm = *((RequestManager **)task->args);
  TreeVerifyBatchConfig const &bc =
      Future(task->futures[0]).get_result<TreeVerifyBatchConfig>();
  InferenceResult const &result =
      Future(task->futures[1]).get_result<InferenceResult>();
  int model_id = Future(task->futures[2]).get_result<int>();
  return rm->prepare_next_batch_init(bc, result, model_id);
}

BeamSearchBatchConfig
    RequestManager::prepare_next_batch_init(TreeVerifyBatchConfig const &old_bc,
                                            InferenceResult const &result,
                                            int model_id) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  if (verbose) {
    std::cout << "\n############### prepare_next_batch_init ###############\n";
  }

  // Step 1: use result to update requests
  BeamSearchBatchConfig new_bc;
  new_bc.num_tokens = 0;
  new_bc.model_id = model_id;
  int result_index = 0;

  int num_generation_tokens = 0;
  int num_active_req = -1;

  for (int i = 0; i < BatchConfig::max_requests_per_batch(); i++) {
    if (old_bc.request_completed[i]) {
      continue;
    }
    size_t guid = old_bc.requestsInfo[i].request_guid;
    Request &request = all_requests[guid];

    std::cout << "[ " << guid << " ]" << std::endl;

    // Verify this: get verified tokens from result
    std::vector<std::pair<BatchConfig::TokenId, int>> tree_outputs =
        std::vector<std::pair<BatchConfig::TokenId, int>>();

    assert(old_bc.num_tokens > 0);

    // reset committed_tokens
    if (committed_tokens.count(guid) == 0) {
      committed_tokens[guid] = {};
    } else {
      committed_tokens[guid].clear();
    }

    // iterate through all the tokens that belong to request i
    int root_abs_depth = request.tokens.size() - 1;

    while (result_index < old_bc.num_tokens &&
           old_bc.tokensInfo[result_index].request_index == i) {
      int abs_depth = old_bc.tokensInfo[result_index].abs_depth_in_request;
      int token_id = result.token_ids[result_index];

      if (request.status == Request::PENDING) {
        committed_tokens[guid].emplace_back(abs_depth, result_index);
      } else if (abs_depth >= root_abs_depth) {
        tree_outputs.emplace_back(token_id, abs_depth + 1);
        // std::cout << "committred tokens push: " << abs_depth
        //           << " ,result index: " << result_index << "\n";
        committed_tokens[guid].emplace_back(abs_depth, result_index);

        if (verbose) {
          std::cout << "Index within old batch: " << result_index << std::endl;
          printf("  Input: [%d] %d ---> [%d] %d \n",
                 abs_depth,
                 old_bc.tokensInfo[result_index].token_id,
                 tree_outputs.back().second,
                 token_id);
        }
        // std::cout << "Index within old batch: " << result_index << std::endl;
        // printf("  Input: [%d] %d ---> [%d] %d \n",
        //        abs_depth,
        //        old_bc.tokensInfo[result_index].token_id,
        //        tree_outputs.back().second,
        //        token_id);
      }
      result_index++;
    }

    if (request.status == Request::RUNNING) {

      std::vector<std::pair<BatchConfig::TokenId, int>> verified_tokens =
          traverse_verify_tree(guid, dfs_tree_inputs.at(guid), tree_outputs);

      log_req_mgr.print("Number of Verified Tokens = %zu",
                        verified_tokens.size());
      // check if the request is finished
      if (verified_tokens.size() + request.tokens.size() >=
          request.max_sequence_length) {
        // Append all verified tokens to the request
        for (auto const &token_pair : verified_tokens) {
          if (token_pair.second < request.max_sequence_length) {
            request.tokens.push_back(token_pair.first);
          }
        }
        log_req_mgr.print("[Done] guid(%zu) with final length(%zu)",
                          request.guid,
                          request.tokens.size());
        std::string output = this->tokenizer_->Decode(request.tokens);
        // Unlike Huggingface, the sentencepiece C++ library automatically
        // removes the BOS token
        if (model_type == ModelType::LLAMA &&
            request.tokens.at(0) == bos_token_id) {
          output = "<s> " + output;
        }
        {
          // update generation result
          GenerationResult &gr = request_generation_results[request.guid];
          assert(gr.guid == request.guid);
          gr.output_tokens = request.tokens;
          gr.output_text = output;
        }
        request.status = Request::COMPLETED;
        trigger_request_completion_future(request.guid);
        log_req_mgr.print("Final output: %s", output.c_str());

        new_bc.request_completed[i] = true;
        new_bc.request_running[i] = false;
        num_processed_requests++;

        // Log profiling info
        ProfileInfo profile_info = profiling_requests[request.guid];
        profile_info.finish_time = Realm::Clock::current_time_in_microseconds();
        profile_info.ssm_decoding_steps = 0;
        total_request_run_time +=
            profile_info.finish_time - profile_info.start_time;
        profiling_requests[request.guid] = profile_info;
        log_req_mgr.print(
            "[Profile] guid(%zu) llm_decoding_steps(%d) start(%.1lf) "
            "finish(%.1lf) latency(%.1lf)",
            request.guid,
            profile_info.llm_decoding_steps,
            profile_info.start_time,
            profile_info.finish_time,
            profile_info.finish_time - profile_info.start_time);

        // Write output to file if needed:
        if (!output_filepath.empty()) {
          std::ofstream outputFile(output_filepath, std::ios::app);
          if (outputFile.is_open()) {
            outputFile << "end-to-end latency: " << std::fixed
                       << std::setprecision(3) << total_request_run_time
                       << std::endl;
            outputFile << "num decoding steps: "
                       << profile_info.llm_decoding_steps << std::endl;
            outputFile << "token IDs: ";
            for (int i = 0; i < request.tokens.size(); i++) {
              outputFile << request.tokens[i];
              if (i < request.tokens.size() - 1) {
                outputFile << ",";
              }
            }
            outputFile << std::endl;
            outputFile << output;

            outputFile.close();
          } else {
            std::cout << "Unable to open the output file: " << output_filepath
                      << std::endl;
            assert(false);
          }
        }

        // delete the old input tree from cache
        dfs_tree_inputs.erase(request.guid);

      } else { // Request not finished, pass verified_tokens to next iteration

        new_bc.request_completed[i] = false;
        new_bc.request_running[i] = true;
        num_active_req++;

        // Normal Request Info
        new_bc.requestsInfo[i].first_token_depth_in_request =
            verified_tokens.front().second;
        new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
        new_bc.requestsInfo[i].request_guid =
            old_bc.requestsInfo[i].request_guid;
        new_bc.requestsInfo[i].max_sequence_length =
            old_bc.requestsInfo[i].max_sequence_length;
        new_bc.requestsInfo[i].num_tokens_in_batch = verified_tokens.size();
        new_bc.requestsInfo[num_active_req].batch_config_request_id = i;

        // TODO: Beam Request Info, missing from VerifyTreeBatchConfig
        int new_max_depth =
            new_bc.requestsInfo[i].max_sequence_length -
            new_bc.requestsInfo[i].first_token_depth_in_request -
            verified_tokens.size();
        new_bc.beamRequestsInfo[i].current_depth = 1;

        profiling_requests[request.guid].ssm_decoding_steps = 0;
        new_bc.requestsInfo[i].prompt_phase = true;

        int ssm_decoding_steps = 0;
        new_bc.beamRequestsInfo[i].beam_size =
            spec_infer_tree_width.size() > ssm_decoding_steps
                ? spec_infer_tree_width[ssm_decoding_steps]
                : 1;
        new_bc.beamRequestsInfo[i].max_depth =
            std::min(new_max_depth, BeamSearchBatchConfig::MAX_BEAM_DEPTH);
        for (int j = 0;
             j < BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
             j++) {
          new_bc.beamRequestsInfo[i].parent_id[j] = 0;
          new_bc.beamRequestsInfo[i].probs[j] = 1;
        }

        new_bc.beamRequestsInfo[i].sub_request_num = 1;

        new_bc.sub_requests[i] = 1;

        updateBitMask(new_bc.causalMask[i],
                      verified_tokens.size(),
                      request.tokens.size());

        // Token Info
        for (int j = 0; j < verified_tokens.size(); j++) {
          auto token = verified_tokens.at(j);

          // Normal Token Info
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].token_id = token.first;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
              token.second;

          // Beam Token Info
          new_bc.beamTokenInfo[new_bc.num_tokens].sub_request_index = 0;
          new_bc.num_tokens++;

          // Add verified token to request's token list
          request.tokens.push_back(token.first);

          if (new_bc.num_tokens == get_max_tokens_per_batch()) {
            break;
          }
        }

        std::string output = this->tokenizer_->Decode(request.tokens);
        // Unlike Huggingface, the sentencepiece C++ library automatically
        // removes the BOS token
        if (model_type == ModelType::LLAMA &&
            request.tokens.at(0) == bos_token_id) {
          output = "<s> " + output;
        }
        log_req_mgr.print("Output: %s", output.c_str());
      }

    } else if (request.status == Request::PENDING) {
      new_bc.request_completed[i] = false;
      new_bc.request_running[i] = false;
      num_active_req++;

      std::cout << "ssm_cache_size: " << request.ssm_cache_size << ", "
                << "initial_len: " << request.initial_len << std::endl;
      assert(request.ssm_cache_size == request.initial_len);

      // Normal Request Info
      new_bc.requestsInfo[i].first_token_depth_in_request =
          request.ssm_cache_size;
      new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
      new_bc.requestsInfo[i].request_guid = old_bc.requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_bc.requestsInfo[i].max_sequence_length;
      new_bc.requestsInfo[i].num_tokens_in_batch = 0;
      new_bc.requestsInfo[num_active_req].batch_config_request_id = i;

      // TODO: Beam Request Info, missing from VerifyTreeBatchConfig
      new_bc.beamRequestsInfo[i].current_depth = 1;
      int ssm_decoding_steps =
          profiling_requests[request.guid].ssm_decoding_steps;
      new_bc.beamRequestsInfo[i].beam_size =
          spec_infer_tree_width.size() > ssm_decoding_steps
              ? spec_infer_tree_width[ssm_decoding_steps]
              : 1;
      new_bc.beamRequestsInfo[i].max_depth = 0;
      for (int j = 0; j < BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
           j++) {
        new_bc.beamRequestsInfo[i].parent_id[j] = 0;
        new_bc.beamRequestsInfo[i].probs[j] = 1;
      }

      new_bc.beamRequestsInfo[i].sub_request_num = 1;

      new_bc.sub_requests[i] = 1;

      // Token Info
      std::string output = this->tokenizer_->Decode(request.tokens);
      // Unlike Huggingface, the sentencepiece C++ library automatically removes
      // the BOS token
      if (model_type == ModelType::LLAMA &&
          request.tokens.at(0) == bos_token_id) {
        output = "<s> " + output;
      }
      log_req_mgr.print("Output: %s", output.c_str());
    } else {
      assert(false);
    }
  }

  // Step 2: Initialize new request
  for (int i = 0; i < BeamSearchBatchConfig::max_requests_per_batch(); i++) {
    if (new_bc.request_completed[i]) {
      if (!pending_request_queue.empty() &&
          new_bc.num_tokens < get_max_tokens_per_batch()) {
        Request new_request = pending_request_queue.front();
        pending_request_queue.pop();
        // all_requests[new_request.guid] = new_request;
        num_active_req++;
        new_bc.requestsInfo[i].first_token_depth_in_request = 0;
        new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
        new_bc.requestsInfo[i].request_guid = new_request.guid;
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(get_max_tokens_per_batch() - new_bc.num_tokens,
                     (int)new_request.tokens.size());
        new_bc.requestsInfo[i].max_sequence_length =
            new_request.max_sequence_length;
        new_bc.requestsInfo[num_active_req].batch_config_request_id = i;

        // add profile_info for the new request
        ProfileInfo profile_info;
        profile_info.llm_decoding_steps = 0;
        profile_info.ssm_decoding_steps = 0;
        profile_info.start_time = Realm::Clock::current_time_in_microseconds();
        profiling_requests[new_request.guid] = profile_info;
        // init the beam search metadata per request
        int ssm_decoding_steps = profile_info.ssm_decoding_steps;

        new_bc.beamRequestsInfo[i].beam_size =
            spec_infer_tree_width.size() > ssm_decoding_steps
                ? spec_infer_tree_width[ssm_decoding_steps]
                : 1;
        new_bc.beamRequestsInfo[i].current_depth = 1;
        new_bc.beamRequestsInfo[i].max_depth =
            std::min(BeamSearchBatchConfig::MAX_BEAM_DEPTH,
                     get_max_tokens_per_batch() -
                         new_bc.requestsInfo[i].num_tokens_in_batch - 1);
        for (int j = 0;
             j < BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES;
             j++) {
          new_bc.beamRequestsInfo[i].parent_id[j] = 0;
          new_bc.beamRequestsInfo[i].probs[j] = 1;
        }

        new_bc.request_completed[i] = false;
        new_bc.requestsInfo[i].prompt_phase = true;

        new_bc.beamRequestsInfo[i].sub_request_num = 1;
        printf("sub request num == 1, %d \n",
               new_bc.beamRequestsInfo[i].beam_size);

        new_bc.sub_requests[i] = 1;

        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          int depth = new_bc.requestsInfo[i].first_token_depth_in_request + j;
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

        initBitMask(new_bc.causalMask[i],
                    new_bc.requestsInfo[i].num_tokens_in_batch);

        // if (new_bc.requestsInfo[i].num_tokens_in_batch <
        // new_request.initial_len) {
        //   all_requests[new_request.guid].status = Request::PENDING;
        //   new_bc.request_running[i] = false;
        //   std::cout << "Request " << new_request.guid << " is pending"
        //             << std::endl;
        // } else {
        //   all_requests[new_request.guid].status = Request::RUNNING;
        //   new_bc.request_running[i] = true;
        //   std::cout << "Request " << new_request.guid << " is running"
        //             << std::endl;
        // }
        all_requests[new_request.guid].status = Request::PENDING;
        all_requests[new_request.guid].ssm_cache_size =
            new_bc.requestsInfo[i].num_tokens_in_batch;
        new_bc.request_running[i] = false;
        std::cout << "SSM KV Cache Size init: "
                  << all_requests[new_request.guid].ssm_cache_size << std::endl;
        std::cout << "LLM KV Cache Size init: "
                  << all_requests[new_request.guid].llm_cache_size << std::endl;

        std::cout << "load " << new_bc.requestsInfo[i].num_tokens_in_batch
                  << " tokens for request " << new_request.guid << std::endl;
        std::cout << "total prompt in request: " << new_request.initial_len
                  << std::endl;

        if (new_bc.num_tokens == get_max_tokens_per_batch()) {
          break;
        }
      }
    }
  }
  new_bc.num_generation_tokens = num_generation_tokens;

  if (verbose) {
    std::cout << "prepare_next_batch_init OLD vs NEW batchconfigs below:"
              << std::endl;
    old_bc.print();
    new_bc.print();
  }
  return new_bc;
}

/***** Beam Search Phase *****/
BeamSearchBatchConfigFuture RequestManager::prepare_next_batch_beam(
    BeamSearchBatchConfigFuture const &old_bc,
    BeamInferenceResultFuture const &result,
    Context ctx,
    Runtime *runtime) {

  RequestManager *rm = this;
  TaskLauncher launcher(RM_PREPARE_NEXT_BATCH_BEAM_TASK_ID,
                        TaskArgument(&rm, sizeof(RequestManager *)));
  launcher.add_future(old_bc);
  launcher.add_future(result);
  return runtime->execute_task(ctx, launcher);
}

BeamSearchBatchConfig RequestManager::prepare_next_batch_beam_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  RequestManager *rm = *((RequestManager **)task->args);
  BeamSearchBatchConfig const &bc =
      Future(task->futures[0]).get_result<BeamSearchBatchConfig>();
  BeamInferenceResult const &result =
      Future(task->futures[1]).get_result<BeamInferenceResult>();
  return rm->prepare_next_batch_beam(bc, result);
}

// update beam search metadata
BeamSearchBatchConfig
    RequestManager::prepare_next_batch_beam(BeamSearchBatchConfig const &old_bc,
                                            BeamInferenceResult const &result) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);
  if (verbose) {
    std::cout << "\n############### prepare_next_batch_beam ###############\n";
  }
  if (verbose) {
    std::cout << "print all results"
              << "\n";
    for (int i = 0; i < 40; i++) {
      std::cout << result.token_ids[i] << ", ";
    }
    std::cout << "Current Beam Depth: "
              << old_bc.beamRequestsInfo[0].current_depth << "\n";
    std::cout << "Current sub request num: "
              << old_bc.beamRequestsInfo[0].sub_request_num << "\n";
  }
  // Step 1: Store result to the beam tree struct
  store_beam_metadata(old_bc, result);

  // Step 2: preparing the next batch for existing requests
  BeamSearchBatchConfig new_bc;
  new_bc.model_id = old_bc.model_id;
  // std::cout << "old_bc.model_id: " << old_bc.model_id << "\n";
  int num_generation_tokens = 0;

  // Add incremental tokens to the batch
  int num_active_req = -1;
  for (int i = 0; i < BatchConfig::max_requests_per_batch(); i++) {
    if (old_bc.request_completed[i] || !old_bc.request_running[i]) {
      continue;
    }
    num_active_req++;
    // Comment out this assertion since num_tokens_in_batch can be
    // zero when beam search has reached required sequence length
    // assert(old_bc.requestsInfo[i].num_tokens_in_batch > 0);
    Request &request = all_requests[old_bc.requestsInfo[i].request_guid];
    int processed_tokens = old_bc.requestsInfo[i].first_token_depth_in_request +
                           old_bc.requestsInfo[i].num_tokens_in_batch;

    // assert(processed_tokens < request.tokens.size());
    log_req_mgr.debug() << "processed_tokens: " << processed_tokens << "\n";
    {
      log_req_mgr.debug() << "num tokens: " << old_bc.num_tokens << ", "
                          << new_bc.num_tokens;
      new_bc.request_completed[i] = false;
      new_bc.requestsInfo[i].first_token_depth_in_request = processed_tokens;
      new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
      new_bc.requestsInfo[i].request_guid = old_bc.requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_bc.requestsInfo[i].max_sequence_length;
      profiling_requests[request.guid].ssm_decoding_steps += 1;
      new_bc.requestsInfo[num_active_req].batch_config_request_id = i;
      // update the beam search metadata
      // how many sub request in current request
      // why is sub_requests has max_requests_per_batch() * MAX_BEAM_WIDTH
      // entries?
      // update the parentid, accumalated_probs, depth, and token_ids
      int ssm_decoding_steps =
          profiling_requests[request.guid].ssm_decoding_steps;

      new_bc.beamRequestsInfo[i].beam_size =
          spec_infer_tree_width.size() > ssm_decoding_steps
              ? spec_infer_tree_width[ssm_decoding_steps]
              : 1;

      new_bc.beamRequestsInfo[i].max_depth =
          old_bc.beamRequestsInfo[i].max_depth;

      new_bc.sub_requests[i] =
          old_bc.sub_requests[i] * new_bc.beamRequestsInfo[i].beam_size;
      new_bc.beamRequestsInfo[i].sub_request_num =
          old_bc.beamRequestsInfo[i].sub_request_num *
          old_bc.beamRequestsInfo[i].beam_size;

      assert(new_bc.beamRequestsInfo[i].sub_request_num <=
                 BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES &&
             "exceed maximum nodes per layer");

      if (request.status == Request::RUNNING) {
        new_bc.beamRequestsInfo[i].current_depth =
            old_bc.beamRequestsInfo[i].current_depth + 1;
        new_bc.request_running[i] = true;
        // do the slot exchange to minimize the cache exchange in kernel.
        update_beam_metadata(
            new_bc, old_bc, request.beam_trees.at(old_bc.model_id), i);

      } else {
        assert(false && "Request should not be pending in beam search phase");
      }

      // do the slot exchange to minimize the cache exchange in kernel.
      // update_beam_metadata(new_bc, request.beam_trees.at(old_bc.model_id),
      // i);
      if (new_bc.requestsInfo[i].first_token_depth_in_request >=
          request.tokens.size()) {
        // Incremental phase
        if (request.status == Request::RUNNING) {
          // todo this is replaced by this_layer_size, but should check it
          new_bc.requestsInfo[i].num_tokens_in_batch = 1;
        } else {
          assert(false && "Request should be done");
          // new_bc.requestsInfo[i].num_tokens_in_batch = 0;
        }

        if (verbose) {
          std::cout << "[ Beam Spec] " << request.guid << std::endl;
          std::cout << "Incremental phase: " << request.tokens.size()
                    << ", num_tokens_in_batch: "
                    << new_bc.requestsInfo[i].num_tokens_in_batch << std::endl;
        }
      }

      if (verbose) {
        std::cout << "SSM KV Cache Size beam: " << request.ssm_cache_size
                  << std::endl;
        std::cout << "LLM KV Cache Size beam: " << request.llm_cache_size
                  << std::endl;
      }

      // register more tokens due to the beam width

      // copy metadata
      memcpy(&new_bc.causalMask[i],
             &old_bc.causalMask[i],
             sizeof(BatchConfig::BitMask));
      BeamTree tree = request.beam_trees[old_bc.model_id];
      appendBitMask(new_bc.causalMask[i],
                    new_bc.beamRequestsInfo[i].sub_request_num,
                    old_bc.beamRequestsInfo[i].beam_size,
                    old_bc.beamRequestsInfo[i].sub_request_num,
                    tree,
                    old_bc.beamRequestsInfo[i].current_depth);
      for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
        int depth = new_bc.requestsInfo[i].first_token_depth_in_request + j;
        for (int k = 0; k < new_bc.beamRequestsInfo[i].sub_request_num; k++) {
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;

          // get value from requestinfo
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
              new_bc.beamRequestsInfo[i].tokens[k];

          new_bc.beamTokenInfo[new_bc.num_tokens].sub_request_index = k;
          new_bc.num_tokens++;

          num_generation_tokens++;
        }
      }
    }
  }

  // how many requests is in speculative phase
  new_bc.speculative_request_num = num_active_req + 1;

  // Add prompt tokens to the batch
  for (int i = 0; i < BatchConfig::max_requests_per_batch(); i++) {
    if (old_bc.request_completed[i] || old_bc.request_running[i]) {
      continue;
    }
    num_active_req++;
    // Comment out this assertion since num_tokens_in_batch can be
    // zero when beam search has reached required sequence length
    // assert(old_bc.requestsInfo[i].num_tokens_in_batch > 0);
    Request &request = all_requests[old_bc.requestsInfo[i].request_guid];
    int processed_tokens = old_bc.requestsInfo[i].first_token_depth_in_request +
                           old_bc.requestsInfo[i].num_tokens_in_batch;

    // assert(processed_tokens < request.tokens.size());
    log_req_mgr.debug() << "processed_tokens: " << processed_tokens << "\n";

    {
      log_req_mgr.debug() << "num tokens: " << old_bc.num_tokens << ", "
                          << new_bc.num_tokens;
      new_bc.request_completed[i] = false;
      new_bc.requestsInfo[i].first_token_depth_in_request = processed_tokens;
      new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
      new_bc.requestsInfo[i].request_guid = old_bc.requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_bc.requestsInfo[i].max_sequence_length;
      new_bc.requestsInfo[num_active_req].batch_config_request_id = i;

      // update the beam search metadata
      // how many sub request in current request
      // why is sub_requests has max_requests_per_batch() * MAX_BEAM_WIDTH
      // entries?
      int ssm_decoding_steps =
          profiling_requests[request.guid].ssm_decoding_steps;

      new_bc.beamRequestsInfo[i].beam_size = 1;
      // printf("beam size: %d, %d\n",
      //        new_bc.beamRequestsInfo[i].beam_size,
      //        ssm_decoding_steps);
      new_bc.beamRequestsInfo[i].max_depth =
          old_bc.beamRequestsInfo[i].max_depth;
      // new_bc.sub_requests[i] =
      //     old_bc.sub_requests[i] * new_bc.beamRequestsInfo[i].beam_size;
      new_bc.sub_requests[i] = 1;
      new_bc.beamRequestsInfo[i].sub_request_num =
          old_bc.beamRequestsInfo[i].sub_request_num;

      assert(new_bc.beamRequestsInfo[i].sub_request_num <=
                 BeamSearchBatchConfig::MAX_SPECULATIVE_TREE_BRANCHES &&
             "exceed maximum nodes per layer");

      // update the parentid, accumalated_probs, depth, and token_ids

      if (request.status == Request::PENDING) {
        // if the request is pending, we need to update the beam search
        // metadata based on the initial length
        new_bc.beamRequestsInfo[i].current_depth =
            old_bc.beamRequestsInfo[i].current_depth;
        new_bc.request_running[i] = false;
      } else {
        assert(false && "Request should be pending");
      }

      memcpy(&new_bc.causalMask[i],
             &old_bc.causalMask[i],
             sizeof(BatchConfig::BitMask));

      new_bc.requestsInfo[i].prompt_phase = true;
      if (new_bc.requestsInfo[i].first_token_depth_in_request >=
          request.tokens.size()) {
        // request is done
        new_bc.requestsInfo[i].num_tokens_in_batch = 0;
        new_bc.causalMask[i].this_layer_size = 0;
        new_bc.beamRequestsInfo[i].sub_request_num = 0;
        new_bc.beamRequestsInfo[i].beam_size = 1;
      } else {
        // Prompt phase
        new_bc.requestsInfo[i].num_tokens_in_batch =
            std::min(get_max_tokens_per_batch() - new_bc.num_tokens -
                         BatchConfig::max_requests_per_batch() + i,
                     (int)request.tokens.size() -
                         new_bc.requestsInfo[i].first_token_depth_in_request);
        request.ssm_cache_size += new_bc.requestsInfo[i].num_tokens_in_batch;
        BeamTree tree = request.beam_trees[old_bc.model_id];
        appendPendingRequest(new_bc.causalMask[i],
                             new_bc.requestsInfo[i].num_tokens_in_batch);
      }

      if (verbose) {
        std::cout << "[ Beam Spec] " << request.guid << std::endl;
        std::cout << "Prompt phase: " << request.tokens.size()
                  << ", num_tokens_in_batch:"
                  << new_bc.requestsInfo[i].num_tokens_in_batch << std::endl;
        std::cout << "Update ssm cache size: " << request.ssm_cache_size
                  << std::endl;

        std::cout << "SSM KV Cache Size beam: " << request.ssm_cache_size
                  << std::endl;
        std::cout << "LLM KV Cache Size beam: " << request.llm_cache_size
                  << std::endl;
      }

      // register more tokens due to the beam width
      for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
        int depth = new_bc.requestsInfo[i].first_token_depth_in_request + j;
        for (int k = 0; k < new_bc.beamRequestsInfo[i].sub_request_num; k++) {
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request = depth;

          // get value from requestinfo
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
              request.tokens[request.tokens.size() -
                             new_bc.requestsInfo[i].num_tokens_in_batch + j];

          new_bc.beamTokenInfo[new_bc.num_tokens].sub_request_index = k;
          new_bc.num_tokens++;
        }
      }
    }
  }

  new_bc.num_generation_tokens = num_generation_tokens;
  if (verbose) {
    std::cout << "prepare_next_batch_beam OLD vs NEW batchconfigs:"
              << std::endl;
    old_bc.print();
    new_bc.print();
  }
  return new_bc;
}

/***** Verify Phase *****/

TreeVerifyBatchConfigFuture RequestManager::prepare_next_batch_verify(
    std::vector<BeamSearchBatchConfigFuture> const &old_batches,
    Context ctx,
    Runtime *runtime) {

  RequestManager *rm = this;
  TaskLauncher launcher(RM_PREPARE_NEXT_BATCH_VERIFY_TASK_ID,
                        TaskArgument(&rm, sizeof(RequestManager *)));
  for (auto const &bcf : old_batches) {
    launcher.add_future(bcf);
  }
  return runtime->execute_task(ctx, launcher);
}

TreeVerifyBatchConfig RequestManager::prepare_next_batch_verify_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  RequestManager *rm = *((RequestManager **)task->args);
  std::vector<BeamSearchBatchConfig> old_batches;
  for (auto const &bcf : task->futures) {
    old_batches.push_back(Future(bcf).get_result<BeamSearchBatchConfig>());
  }
  return rm->prepare_next_batch_verify(old_batches);
}

TreeVerifyBatchConfig RequestManager::prepare_next_batch_verify(
    std::vector<BeamSearchBatchConfig> const &old_batches) {
  const std::lock_guard<std::mutex> lock(request_queue_mutex);

  if (verbose) {
    std::cout
        << "\n############### prepare_next_batch_verify ###############\n";
  }

  assert(old_batches.size() > 0);

  TreeVerifyBatchConfig new_bc;
  new_bc.num_tokens_to_commit = 0;
  new_bc.num_tokens = 0;

  int max_prompt_load_size = get_max_verify_tokens_per_batch();
  for (int i = 0; i < TreeVerifyBatchConfig::max_requests_per_batch(); i++) {
    if (old_batches.at(0).request_completed[i]) {
      continue;
    } else if (old_batches.at(0).request_running[i]) {
      max_prompt_load_size -= (BeamSearchBatchConfig::MAX_BEAM_DEPTH + 1);
    } else {
      max_prompt_load_size -= 1;
    }
  }
  int num_active_req = -1;
  for (int i = 0; i < TreeVerifyBatchConfig::max_requests_per_batch(); i++) {
    if (old_batches.at(0).request_completed[i]) {
      continue;
    }
    num_active_req++;
    size_t guid = old_batches.at(0).requestsInfo[i].request_guid;
    Request &request = all_requests[guid];

    // Profiling
    profiling_requests[request.guid].llm_decoding_steps += 1;

    if (request.status == Request::RUNNING) {
      new_bc.request_running[i] = true;

      // Get the dfs tree
      std::vector<std::vector<std::pair<BatchConfig::TokenId, int>>>
          all_dfs_trees;

      for (int j = 0; j < old_batches.size(); j++) {
        std::vector<std::pair<BatchConfig::TokenId, int>> new_tree =
            traverse_beam_tree(old_batches.at(j), i, request.tokens.size() - 1);
        all_dfs_trees.push_back(new_tree);
      }
      assert(all_dfs_trees.size() == old_batches.size());
      std::vector<std::pair<BatchConfig::TokenId, int>> dfs_tree_inputs =
          merge_dfs_trees(all_dfs_trees, request.tokens.size() - 1, guid);

      if (verbose) {
        std::cout << "Request Tokens Size: " << request.tokens.size()
                  << std::endl;
        for (int k = 0; k < request.tokens.size(); k++) {
          std::cout << k << ": " << request.tokens[k] << std::endl;
        }
      }

      // Normal Request Info
      new_bc.requestsInfo[i].first_token_depth_in_request =
          dfs_tree_inputs.front().second;
      new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
      new_bc.requestsInfo[i].request_guid =
          old_batches.at(0).requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_batches.at(0).requestsInfo[i].max_sequence_length;
      new_bc.requestsInfo[num_active_req].batch_config_request_id = i;

      // copy bitmask to verify batchconfig
      memcpy(&(new_bc.causalMask[i]),
             &(old_batches.at(0).causalMask[i]),
             sizeof(BatchConfig::BitMask));
      // TODO: Check this
      new_bc.requestsInfo[i].num_tokens_in_batch = 0;
      new_bc.request_completed[i] = false;

      // std::cout << "dfs_tree_inputs: " << dfs_tree_inputs.size() << ", "
      //           << new_bc.causalMask[i].tree_size << ", "
      //           << new_bc.causalMask[i].non_tree_cache_size << "\n";
      // std::cout << "mask: " << std::bitset<64>(new_bc.causalMask[i].mask[0])
      //           << "\n";

      // Committed Tokens
      if (committed_tokens.find(guid) != committed_tokens.end()) {
        for (int j = 0; j < committed_tokens.at(guid).size(); j++) {
          // if (j < committed_tokens.at(guid).size()) {

          auto committed_token = committed_tokens.at(guid).at(j);
          new_bc.committed_tokens[new_bc.num_tokens_to_commit].token_index =
              committed_token.second;
          new_bc.committed_tokens[new_bc.num_tokens_to_commit].request_index =
              i;
          new_bc.committed_tokens[new_bc.num_tokens_to_commit].token_depth =
              committed_token.first;
          if (verbose) {
            std::cout << new_bc.num_tokens_to_commit
                      << "- committed_token.token_depth: "
                      << committed_token.first
                      << ", token_index: " << committed_token.second
                      << std::endl;
          }
          new_bc.num_tokens_to_commit++;
          request.llm_cache_size++;
          // }
        }
      }
      if (verbose) {
        std::cout << "new_bc.num_tokens_to_commit: "
                  << new_bc.num_tokens_to_commit << std::endl;
      }

      // Incremental phase: only add the last committed token
      new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
      new_bc.tokensInfo[new_bc.num_tokens].token_id = request.tokens.back();
      new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
          request.tokens.size() - 1;

      new_bc.num_tokens++;
      new_bc.requestsInfo[i].num_tokens_in_batch++;

      if (new_bc.num_tokens > get_max_verify_tokens_per_batch()) {
        assert(false &&
               "Exceeding the space available in the TreeVerify batch");
        break;
      }

      new_bc.requestsInfo[i].first_token_depth_in_request =
          request.tokens.size() - 1;

      bool cutLayer = false;
      // Add Tokens from the DFS Tree to the next batch
      for (int j = 1; j < dfs_tree_inputs.size(); j++) {
        auto token = dfs_tree_inputs.at(j);
        if (verbose) {
          std::cout << "[" << j << "] Token: " << token.first
                    << ", Depth:" << token.second << std::endl;
        }
        // Normal Token Info
        new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
        new_bc.tokensInfo[new_bc.num_tokens].token_id = token.first;
        new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
            token.second;

        new_bc.num_tokens++;
        new_bc.requestsInfo[i].num_tokens_in_batch++;

        if (new_bc.num_tokens == get_max_verify_tokens_per_batch() &&
            (j != dfs_tree_inputs.size() - 1)) {
          cutLayer = true;
          break;
        }
      }

      // delete the last incomplete layer
      if (cutLayer) {
        int total_tokens = new_bc.num_tokens;
        for (int j = total_tokens - 1; j >= 1; j--) {
          new_bc.num_tokens--;
          new_bc.requestsInfo[i].num_tokens_in_batch--;
          // std::cout << "cut: " << j << "\n";
          if (new_bc.tokensInfo[j].abs_depth_in_request !=
              new_bc.tokensInfo[j - 1].abs_depth_in_request) {
            break;
          }
        }
      }

    } else if (request.status == Request::PENDING) {
      new_bc.request_running[i] = false;
      if (verbose) {
        std::cout << "[Verify] Request " << request.guid
                  << " is pending in loading prompt phase" << std::endl;
        std::cout << "SSM KV Cache Size verify: " << request.ssm_cache_size
                  << std::endl;
        std::cout << "LLM KV Cache Size verify: " << request.llm_cache_size
                  << std::endl;
      }

      // Commit all tokens from the last loading batch
      if (committed_tokens.find(guid) != committed_tokens.end()) {
        for (int j = 0; j < committed_tokens.at(guid).size(); j++) {
          auto token = committed_tokens.at(guid).at(j);
          new_bc.committed_tokens[new_bc.num_tokens_to_commit].token_index =
              token.second;
          new_bc.committed_tokens[new_bc.num_tokens_to_commit].request_index =
              i;
          new_bc.committed_tokens[new_bc.num_tokens_to_commit].token_depth =
              token.first;

          new_bc.num_tokens_to_commit++;
          request.llm_cache_size++;
        }
        std::cout << "[Verify] Committed Tokens from last loading batch: "
                  << new_bc.num_tokens_to_commit << std::endl;
      }

      memcpy(&(new_bc.causalMask[i]),
             &(old_batches.at(0).causalMask[i]),
             sizeof(BatchConfig::BitMask));

      // Normal Request Info
      new_bc.requestsInfo[i].first_token_depth_in_request =
          request.llm_cache_size;
      new_bc.requestsInfo[i].first_token_offset_in_batch = new_bc.num_tokens;
      new_bc.requestsInfo[i].request_guid =
          old_batches.at(0).requestsInfo[i].request_guid;
      new_bc.requestsInfo[i].max_sequence_length =
          old_batches.at(0).requestsInfo[i].max_sequence_length;
      new_bc.requestsInfo[num_active_req].batch_config_request_id = i;

      new_bc.request_completed[i] = false;

      new_bc.requestsInfo[i].num_tokens_in_batch =
          std::min(max_prompt_load_size,
                   (int)request.initial_len -
                       new_bc.requestsInfo[i].first_token_depth_in_request);
      max_prompt_load_size -= new_bc.requestsInfo[i].num_tokens_in_batch;

      std::cout << "max_prompt_load_size: " << max_prompt_load_size
                << std::endl;

      if (request.llm_cache_size < request.initial_len) {
        // std::cout << "Initialization (prompt) phase: "
        //           << new_bc.requestsInfo[i].num_tokens_in_batch << ", "
        //           << old_batches.at(0).beamRequestsInfo[i].beam_size << "\n";
        // Initialization (prompt) phase
        for (int j = 0; j < new_bc.requestsInfo[i].num_tokens_in_batch; j++) {
          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].token_id =
              request.tokens[request.llm_cache_size + j];
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
              request.llm_cache_size + j;
          new_bc.num_tokens++;
        }

        if (new_bc.num_tokens > get_max_verify_tokens_per_batch()) {
          printf("Exceeding (%i) the space available (%i) in the TreeVerify "
                 "batch\n",
                 new_bc.num_tokens,
                 get_max_verify_tokens_per_batch());
          assert(false);
        }

        if (new_bc.requestsInfo[i].num_tokens_in_batch +
                request.llm_cache_size >=
            request.initial_len) {
          // launch the request into running phase after loading all prompt
          request.status = Request::RUNNING;
          new_bc.request_running[i] = true;

          // std::cout << "new_bc.requestsInfo[i].num_tokens_in_batch: "
          //           << new_bc.requestsInfo[i].num_tokens_in_batch <<
          //           std::endl;
          new_bc.requestsInfo[i].prompt_phase = true;

          dfs_tree_inputs[guid] =
              std::vector<std::pair<BatchConfig::TokenId, int>>{std::make_pair(
                  request.tokens.back(), request.tokens.size() - 1)};
        }
      } else { // launch the request into running phase after loading all prompt
        if (get_max_verify_tokens_per_batch() - new_bc.num_tokens > 0) {
          // std::cout << "Initialization running phase: "
          //           << new_bc.requestsInfo[i].num_tokens_in_batch << "\n";
          request.status = Request::RUNNING;
          new_bc.request_running[i] = true;

          new_bc.tokensInfo[new_bc.num_tokens].request_index = i;
          new_bc.tokensInfo[new_bc.num_tokens].token_id = request.tokens.back();
          new_bc.tokensInfo[new_bc.num_tokens].abs_depth_in_request =
              request.tokens.size() - 1;

          new_bc.num_tokens++;
          new_bc.requestsInfo[i].num_tokens_in_batch++;
          // std::cout << "new_bc.requestsInfo[i].num_tokens_in_batch2: "
          //           << new_bc.requestsInfo[i].num_tokens_in_batch <<
          //           std::endl;

          new_bc.requestsInfo[i].prompt_phase = true;
          dfs_tree_inputs[guid] =
              std::vector<std::pair<BatchConfig::TokenId, int>>{std::make_pair(
                  request.tokens.back(), request.tokens.size() - 1)};
        }
      }

    } else {
      assert(false && "Request status is not RUNNING or PENDING");
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
  auto start_depth = old_bc.tokensInfo[0].abs_depth_in_request;
  int result_index = 0;

  if (verbose) {
    std::cout << "Store total of " << old_bc.num_tokens
              << " tokens in the current batch.\n";
  }

  for (int i = 0; i <= old_bc.num_tokens; i++) {
    if (i == old_bc.num_tokens ||
        old_bc.requestsInfo[old_bc.tokensInfo[i].request_index].request_guid !=
            guid) {

      // std::cout << "i is: " << i << "old guid" << guid << " new guid"
      //           << old_bc.requestsInfo[old_bc.tokensInfo[i].request_index]
      //                  .request_guid
      //           << "\n";

      int index = old_bc.tokensInfo[i - 1].request_index;
      int beam_size = old_bc.beamRequestsInfo[index].beam_size;

      // int leaf_node_num = old_bc.sub_requests[index];
      int leaf_node_num =
          old_bc.beamRequestsInfo[index].sub_request_num * beam_size;
      int depth = old_bc.beamRequestsInfo[index].current_depth;

      // Each token yields (beam_width) results
      // int beam_width = old_bc.beamRequestsInfo[index].beam_size;

      // Count tokens sent to model in this request to find the final token's
      // index
      result_index +=
          (old_bc.tokensInfo[i - 1].abs_depth_in_request - start_depth) *
          beam_size;

      if (verbose) {
        std::cout << "i = " << i << ", result index = " << result_index
                  << ", value: " << result.token_ids[result_index]
                  << ", leaf node num: " << leaf_node_num << ", depth" << depth
                  << ", beam size: " << beam_size << "\n";
      }

      Request &request = all_requests[old_bc.requestsInfo[index].request_guid];

      if (old_bc.requestsInfo[index].num_tokens_in_batch == 0) {
        continue;
      }

      if (depth == 1) {
        // store the last input into the tree;
        if (verbose) {
          std::cout << "try to store the input"
                    << "\n";
        }

        request.beam_trees.at(old_bc.model_id).treeLayers[0].tokens[0] =
            request.tokens.back();
        request.beam_trees.at(old_bc.model_id).treeLayers[0].probs[0] = 1;
        request.beam_trees.at(old_bc.model_id).treeLayers[0].parent_ids[0] = -1;
        request.beam_trees.at(old_bc.model_id)
            .treeLayers[0]
            .nodes_num_this_layer = 1;

        if (verbose) {
          std::cout << "Store the previous last token to the tree root: "
                    << request.tokens.back() << "\n";
        }
      }
      request.beam_trees.at(old_bc.model_id)
          .treeLayers[depth]
          .nodes_num_this_layer = leaf_node_num;
      for (int beam_id = 0; beam_id < leaf_node_num; beam_id++) {

        request.beam_trees.at(old_bc.model_id)
            .treeLayers[depth]
            .tokens[beam_id] = result.token_ids[result_index];
        request.beam_trees.at(old_bc.model_id)
            .treeLayers[depth]
            .probs[beam_id] = result.probs[result_index];
        request.beam_trees.at(old_bc.model_id)
            .treeLayers[depth]
            .parent_ids[beam_id] = result.parent_id[result_index];

        if (verbose) {
          std::cout << "tree value: " << depth << "token: "
                    << request.beam_trees.at(old_bc.model_id)
                           .treeLayers[depth]
                           .tokens[beam_id]
                    << "result tokens: " << result.token_ids[result_index];
        }
        result_index += 1;
      }
      // update the guid and start_depth for current request
      if (i < old_bc.num_tokens) {
        int new_req_idx = old_bc.tokensInfo[i].request_index;
        guid = old_bc.requestsInfo[new_req_idx].request_guid;
        start_depth = old_bc.tokensInfo[i].abs_depth_in_request;
      }
    }
  }
}

// for updating the beam search metadata in requests in incremental phase
void RequestManager::update_beam_metadata(BeamSearchBatchConfig &new_bc,
                                          BeamSearchBatchConfig const &old_bc,
                                          BeamTree &tree,
                                          int request_index) {

  // do the exchange
  if (new_bc.request_completed[request_index]) {
    assert(false);
  }
  int depth = new_bc.beamRequestsInfo[request_index].current_depth - 1;
  int beam_size = new_bc.beamRequestsInfo[request_index].beam_size;

  // int leaf_node_num = old_bc.sub_requests[request_index];
  int leaf_node_num = new_bc.beamRequestsInfo[request_index].sub_request_num;

  if (new_bc.beamRequestsInfo[request_index].current_depth ==
      1) { // TODO: check if this is correct
    // for (int j = 0; j < beam_size; j++) {
    //   new_bc.beamRequestsInfo[request_index].parent_id[j] = j;
    //   new_bc.beamRequestsInfo[request_index].probs[j] =
    //       tree.treeLayers[depth].probs[j]; // ?
    //   new_bc.beamRequestsInfo[request_index].tokens[j] =
    //       tree.treeLayers[depth].tokens[j]; // ?
    // }
    // Do nothing
    // assert(false);
  } else {
    for (int j = 0; j < leaf_node_num; j++) {
      new_bc.beamRequestsInfo[request_index].parent_id[j] =
          tree.treeLayers[depth].parent_ids[j];
      new_bc.beamRequestsInfo[request_index].probs[j] =
          tree.treeLayers[depth].probs[j];
      new_bc.beamRequestsInfo[request_index].tokens[j] =
          tree.treeLayers[depth].tokens[j];
      // std::cout << "token: " << j << ": "
      //           << new_bc.beamRequestsInfo[request_index].tokens[j] << "\n";
    }
  }
  if (verbose) {
    std::cout << "-----------after parent id exchange-----------" << std::endl;
    for (int j = 0; j < beam_size; j++) {
      std::cout << "after request id: " << request_index << "beam id = " << j
                << "parent: "
                << new_bc.beamRequestsInfo[request_index].parent_id[j]
                << "token: " << new_bc.beamRequestsInfo[request_index].tokens[j]
                << "probs: " << new_bc.beamRequestsInfo[request_index].probs[j]
                << std::endl;
    }
  }
}

// bit mask related function

// prompt phase, init task
void RequestManager::initBitMask(BatchConfig::BitMask &bitmask,
                                 int initLength) {
  assert(initLength > 0);
  // eg. 4 tokens: t1: 0000000..1111, t2: 0000000..1110, t3: 0000000..1100, t4:
  // 0000000..1000
  bitmask.non_tree_cache_size = 0;
  bitmask.tree_size = 1;

  bitmask.prompt_size = initLength;
  bitmask.this_layer_size = initLength;
  // std::cout << "see bit mask" << bitmask.prompt_size << "\n";
  // std::cout << "see bit mask" << std::bitset<64>(bitmask.mask[0]) << "\n";
  // std::cout << "see bit mask" << std::bitset<64>(bitmask.mask[1]) << "\n";
  // std::cout << "see bit mask" << std::bitset<64>(bitmask.mask[2]) << "\n";
}

// prepare next init
void RequestManager::updateBitMask(BatchConfig::BitMask &bitmask,
                                   int initLength,
                                   int non_tree_size) {
  // assert(initLength == 1);
  // eg. 4 tokens: t1: 0000000..1111, t2: 0000000..1110, t3: 0000000..1100, t4:
  // 0000000..1000
  assert(initLength <= BatchConfig::MAX_SPEC_TREE_TOKEN_NUM &&
         "do not support tree size > 64");
  assert(initLength >= 1 && "verified token num should >= 1");

  // std::cout << "non tree size: " << non_tree_size << ", "
  //           << bitmask.non_tree_cache_size << "\n";

  bitmask.non_tree_cache_size = non_tree_size + initLength - 1;
  bitmask.tree_size = 1;
  bitmask.this_layer_size = initLength;
  // std::cout << "non_tree_size: " << non_tree_size << "\n";
  bitmask.prompt_size = 1;
  for (int i = 0; i < bitmask.prompt_size; i++) {
    for (int j = i; j < bitmask.prompt_size; j++) {
      bitmask.mask[i] |= (1 << j);
    }
  }

  // std::cout << "see bit mask update" << bitmask.prompt_size << "\n";
  // std::cout << "see bit mask update" << std::bitset<64>(bitmask.mask[0])
  //           << "\n";
}

// prompt phase, init task
void RequestManager::appendPendingRequest(BatchConfig::BitMask &bitmask,
                                          int initLength) {
  assert(initLength > 0);
  // std::cout << "append pending bit mask: " << initLength << "\n";
  // eg. 4 tokens: t1: 0000000..1111, t2: 0000000..1110, t3: 0000000..1100, t4:
  // 0000000..1000
  bitmask.non_tree_cache_size = 0;
  bitmask.tree_size = 1;
  bitmask.prompt_size += initLength;
  bitmask.this_layer_size = initLength;

  // for (int i = 0; i < bitmask.prompt_size; i++) {
  //   for (int j = i; j < bitmask.prompt_size; j++) {
  //     bitmask.mask[i] |= (1 << j);
  //   }
  // }
}

// prepare next beam, append layers to the tree
void RequestManager::appendBitMask(BatchConfig::BitMask &bitmask,
                                   int newNodes,
                                   int preBeamSize,
                                   int old_sub_num,
                                   BeamTree const tree,
                                   int currentDepth) {
  int pre_tree_size = bitmask.tree_size;
  bitmask.tree_size += newNodes;
  bitmask.this_layer_size = newNodes;
  assert(bitmask.tree_size <= BatchConfig::MAX_SPEC_TREE_TOKEN_NUM &&
         "do not support tree size > 64");
  // preBeamSize: replicate num

  // add relationship with input/prompt
  for (int i = 0; i < bitmask.prompt_size; i++) {
    for (int j = pre_tree_size; j < bitmask.tree_size; j++) {
      bitmask.mask[i] |= (1 << j);
      // std::cout << "see bit mask append: " << i << ", to" << j
      //           << std::bitset<64>(bitmask.mask[i]) << "\n";
    }
  }

  // std::cout << "bitmask.tree_size: " << bitmask.tree_size << ", "
  //           << pre_tree_size << ", " << bitmask.prompt_size << ", "
  //           << preBeamSize << "\n";

  // int num_groups = newNodes / preBeamSize;
  // int group_size = newNodes / num_groups;
  // add relations to branch
  // requests in same groups share same relations, except the last token.

  // set middle layers
  //  skip the root prompt/tokens
  int token_idx = bitmask.prompt_size;
  int new_nodes_start_idx = pre_tree_size;
  // std::cout << "new nodes start " << new_nodes_start_idx << "\n";
  for (int i = 1; i < currentDepth; i++) {
    new_nodes_start_idx = pre_tree_size;
    int nodes_this_layer = tree.treeLayers[i].nodes_num_this_layer;
    // std::cout << "tree layer: " << i << " nodes:" << nodes_this_layer
    //           << "group size: " << newNodes / nodes_this_layer << "\n";
    for (int j = 0; j < nodes_this_layer; j++) {
      int group_size = newNodes / nodes_this_layer;
      for (int k = 0; k < group_size; k++) {
        bitmask.mask[token_idx] |= (1 << new_nodes_start_idx);
        new_nodes_start_idx += 1;
      }
      token_idx += 1;
    }
  }

  assert(token_idx == pre_tree_size);
  assert(currentDepth <= 1 || new_nodes_start_idx == bitmask.tree_size);

  // assert(currentDepth <= 2);
  // set last layer, all tokens are only relevant to it self;
  for (int i = token_idx; i < bitmask.tree_size; i++) {
    bitmask.mask[i] |= (1 << i);
    // std::cout << "set rel: " << i << "to: " << i << "\n";
  }

  // if(bitmask.non_tree_cache_size == 19 && bitmask.tree_size > 2){
  //   assert(false);
  // }

  // std::cout << "see bit mask append" << bitmask.prompt_size << "\n";
  // std::cout << "see bit mask append" << bitmask.non_tree_cache_size << "\n";
  // std::cout << "see bit mask append" << std::bitset<64>(bitmask.mask[0])
  //           << "\n";
}

bool PreOrder(
    BeamTree const &tree,
    int max_depth,
    int current_depth,
    int beam_width,
    int id,
    std::vector<std::pair<BeamSearchBatchConfig::TokenId, int>> &serializedTree,
    bool verbose) {
  // terminate
  if (current_depth >= max_depth) {
    serializedTree.push_back(std::make_pair(
        tree.treeLayers[current_depth].tokens[id], current_depth));
    if (verbose) {
      std::cout << "last tokens: " << tree.treeLayers[current_depth].tokens[id]
                << "\n";
      std::cout << "return true"
                << "\n";
    }
    return true;
  }

  // add to tree;
  // std::cout<<"node: " << current_depth << ", id: " <<
  serializedTree.push_back(
      std::make_pair(tree.treeLayers[current_depth].tokens[id], current_depth));
  if (verbose) {
    std::cout << "push something: " << tree.treeLayers[current_depth].tokens[id]
              << ", " << current_depth << std::endl;
  }
  int index = serializedTree.size() - 1;
  int next_layers = current_depth + 1;

  bool flag = false;
  // recursion
  for (int i = 0; i < beam_width; i++) {
    int child_id = i;
    int child_parent = tree.treeLayers[next_layers].parent_ids[i];

    // for all childs, do preOrder
    if (child_parent == id) {
      if (verbose) {
        std::cout << "current depth: " << current_depth << ", child_parent, "
                  << child_parent << ", child_id, " << child_id << "\n";
      }
      bool res = PreOrder(tree,
                          max_depth,
                          current_depth + 1,
                          beam_width,
                          child_id,
                          serializedTree,
                          verbose);
      flag = flag || res;
    }
  }
  // if (!flag) {
  //   // no child for this token, delete it
  //   std::cout << "delete a node: " <<
  //   tree.treeLayers[current_depth].tokens[id]
  //             << ", " << current_depth << std::endl;
  //   serializedTree.erase(serializedTree.begin() + index);
  // }
  return flag;
}

std::vector<std::pair<BatchConfig::TokenId, int>>
    RequestManager::traverse_verify_tree(
        size_t guid,
        std::vector<std::pair<BatchConfig::TokenId, int>> const
            &inputSerializedTree,
        std::vector<std::pair<BatchConfig::TokenId, int>> const
            &outputSerializedTree) {
  std::vector<std::pair<BeamSearchBatchConfig::TokenId, int>> verifiedTree;
  // verifiedTree.push_back(inputSerializedTree.at(0));
  std::vector<std::pair<int, int>> new_committed_tokens =
      std::vector<std::pair<int, int>>();

  log_req_mgr.print("Input tree size (%zu) Output tree size (%zu)",
                    inputSerializedTree.size(),
                    outputSerializedTree.size());
  { // Input tree
    std::ostringstream oss;
    // inputSerializedTree is the dfs_tree_inputs_map[guid] array og (token id,
    // depth) pairs
    for (auto const &pair : inputSerializedTree) {
      oss << " " << pair.second << ":" << pair.first;
      // log_req_mgr.print("(%d, %d)", pair.first, pair.second);
    }
    log_req_mgr.print("Input tree:%s", oss.str().c_str());
  }
  { // Output tree
    // log_req_mgr.print("========Output============");
    // outputSerializedTree is an array of (token id, depth + 1) pairs
    std::ostringstream oss;
    for (auto const &pair : outputSerializedTree) {
      // log_req_mgr.print("(%d, %d)", pair.first, pair.second);
      oss << " " << pair.second << ":" << pair.first;
    }
    log_req_mgr.print("Output tree:%s", oss.str().c_str());
  }
  {
    // log_req_mgr.print("========Committed============");
    //  committed_tokens[guid] is an array of (depth, result_index) pairs for
    //  the given request
    std::ostringstream oss;
    for (auto const &pair : committed_tokens.at(guid)) {
      // log_req_mgr.print("(%d, %d)", pair.first, pair.second);
      oss << " " << pair.second << ":" << pair.first;
    }
    log_req_mgr.print("Committed tokens:%s", oss.str().c_str());
  }

  // It's safe to have inputSerializedTree.size() > outputSerializedTree.size()
  // In this case the inputSeriedTree ends with padding 0s
  assert(inputSerializedTree.size() >= outputSerializedTree.size());

  int *treeLayers = new int[inputSerializedTree.size()];
  int node_num = 1;
  int layer_num = 0;
  for (int token_id = 0; token_id < inputSerializedTree.size(); token_id++) {
    if (token_id == (inputSerializedTree.size() - 1) ||
        inputSerializedTree.at(token_id + 1).second !=
            inputSerializedTree.at(token_id).second) {
      treeLayers[layer_num] = node_num;
      layer_num += 1;
      node_num = 1;
    } else {
      node_num++;
    }
  }

  // to avoid branch switch when same tokens in input tree.
  // todo, only checked for N->1->1->1 cases

  bool findFirst = false;
  layer_num = -1;
  int first_layer_slot = 0;
  int first_layer_slot_total = 0;
  int processed_whole_layer_tokens = 0;

  for (int i = 0; i < outputSerializedTree.size(); i++) {
    auto input = inputSerializedTree.at(i);
    auto output = outputSerializedTree.at(i);

    if (i == 0 || inputSerializedTree.at(i - 1).second !=
                      inputSerializedTree.at(i).second) {
      layer_num += 1;
      processed_whole_layer_tokens += i == 0 ? 0 : treeLayers[layer_num - 1];
    }

    if (i == 0) {
      verifiedTree.push_back(output);

      new_committed_tokens.push_back(std::make_pair(
          input.second,
          committed_tokens.at(guid).at(i).second)); // <input_abs_depth,
                                                    // input_index_in_batch>
      // std::cout << committed_tokens.at(guid).at(i).first << ", "
      //           << committed_tokens.at(guid).at(i).second << std::endl;
      // std::cout << input.first << ", " << input.second << std::endl;

      assert(committed_tokens.at(guid).at(i).first == input.second);
      continue;
    }

    if (input.first == verifiedTree.back().first &&
        input.second == verifiedTree.back().second) {
      if (findFirst) {
        // must in this branch.
        int layer_slot = i - processed_whole_layer_tokens;
        int layer_slot_total = treeLayers[layer_num];
        if ((first_layer_slot == layer_slot)) {
          verifiedTree.push_back(output);
          new_committed_tokens.push_back(std::make_pair(
              input.second, committed_tokens.at(guid).at(i).second));
          // at this point, you'll not go other branches
          // std::cout << "verify tree push back: " << output.first
          //           << ", tree size is: " << verifiedTree.size()
          //           << ", ??: " << input.first << ", " << input.second <<
          //           "\n";

        } else {
          printf("not correct slot\n");
        }
      } else {
        verifiedTree.push_back(output);
        first_layer_slot = i - processed_whole_layer_tokens;
        first_layer_slot_total = treeLayers[layer_num];
        findFirst = true;
        new_committed_tokens.push_back(std::make_pair(
            input.second,
            committed_tokens.at(guid).at(i).second)); // <input_abs_depth,
                                                      // input_index_in_batch>
        // at this point, you'll not go other branches
        // std::cout << "verify tree push back: " << output.first
        //           << ", tree size is: " << verifiedTree.size()
        //           << ", ??: " << input.first << ", " << input.second << "\n";
      }

      assert(committed_tokens.at(guid).at(i).first == input.second);
    }
  }
  committed_tokens[guid] = new_committed_tokens;
  {
    // log_req_mgr.print("========Verified============");
    std::ostringstream oss;
    for (auto const &pair : verifiedTree) {
      // log_req_mgr.print("(%d, %d)", pair.first, pair.second);
      oss << " " << pair.second << ":" << pair.first;
    }
    log_req_mgr.print("Verified:%s", oss.str().c_str());
  }
  {
    // log_req_mgr.print("========New Committed============");
    std::ostringstream oss;
    for (auto const &pair : committed_tokens.at(guid)) {
      // log_req_mgr.print("(%d, %d)", pair.first, pair.second);
      oss << " " << pair.second << ":" << pair.first;
    }
    log_req_mgr.print("New committed:%s", oss.str().c_str());
  }

  return verifiedTree;
}

std::vector<std::pair<BatchConfig::TokenId, int>>
    RequestManager::traverse_beam_tree(BeamSearchBatchConfig const &old_bc,
                                       int request_index,
                                       int first_token_depth_in_request) {
  if (verbose) {
    std::cout << "[Traverse Beam Tree] request_index: " << request_index
              << "\n";
    std::cout << "[Traverse Beam Tree] max_depth: "
              << old_bc.beamRequestsInfo[request_index].max_depth << "\n";
    std::cout << "[Traverse Beam Tree] current_depth: "
              << old_bc.beamRequestsInfo[request_index].current_depth << "\n";
    std::cout << "[Traverse Beam Tree] beam_width: "
              << old_bc.beamRequestsInfo[request_index].beam_size << "\n";
    std::cout << "[Traverse Beam Tree] start index: "
              << first_token_depth_in_request << "\n";
  }

  auto guid = old_bc.requestsInfo[request_index].request_guid;
  Request &request = all_requests[guid];
  // std::cout << "request.beam_trees.size(): " << request.beam_trees.size()
  //           << std::endl;
  BeamTree tree = request.beam_trees.at(old_bc.model_id);

  // std::cout << "print beam tree: "
  //           << "\n";
  std::vector<std::pair<BatchConfig::TokenId, int>> serializedTree;
  for (int i = 0; i <= old_bc.beamRequestsInfo[request_index].max_depth; i++) {
    // std::cout << "tree layer: " << i
    //           << ", num_nodes: " << tree.treeLayers[i].nodes_num_this_layer
    //           << "\n";
    // push tokens into tree
    for (int j = 0; j < tree.treeLayers[i].nodes_num_this_layer; j++) {
      // std::cout << "token: " << tree.treeLayers[i].tokens[j] << "\n";
      serializedTree.push_back(std::make_pair(tree.treeLayers[i].tokens[j], i));
    }
  }
  // token, index
  // todo make this one global for different stages

  // PreOrder(tree,
  //          old_bc.beamRequestsInfo[request_index].max_depth,
  //          0,
  //          old_bc.beamRequestsInfo[request_index].beam_size,
  //          0,
  //          serializedTree,
  //          verbose);

  // print it
  if (verbose) {
    std::cout << "Print serialized tree: size:" << request_index
              << serializedTree.size() << "\n";
  }
  for (int k = 0; k < serializedTree.size(); k++) {
    serializedTree.at(k).second += first_token_depth_in_request;
    if (verbose) {
      std::cout << "token id: " << serializedTree.at(k).first
                << ", depth: " << serializedTree.at(k).second << "\n";
    }
  }

  // if (dfs_tree_inputs.find(old_bc.requestsInfo[request_index].request_guid)
  // !=
  //     dfs_tree_inputs.end()) {
  //   dfs_tree_inputs[old_bc.requestsInfo[request_index].request_guid] =
  //       serializedTree;
  // } else {
  //   dfs_tree_inputs.insert(std::make_pair(
  //       old_bc.requestsInfo[request_index].request_guid, serializedTree));
  // }

  return serializedTree;
  // }
}

std::vector<std::pair<BatchConfig::TokenId, int>>
    RequestManager::merge_dfs_trees(
        std::vector<std::vector<std::pair<BatchConfig::TokenId, int>>>
            input_trees,
        int root_depth,
        RequestGuid guid) {
  assert(input_trees.size() == 1 && "currently using one ssm");
  dfs_tree_inputs[guid] = input_trees.at(0);
  return input_trees.at(0);

  std::vector<std::pair<BatchConfig::TokenId, int>> merged_tree;

  std::unordered_map<int, std::set<int>> childrens;
  std::unordered_map<int, int> curr_path;

  // convert <token_id, depth> pair to an integer
  auto root = input_trees.at(0).at(0);
  int root_id = root.first * 10000 + root.second;

  for (int i = 0; i < input_trees.size(); i++) {
    auto tree = input_trees.at(i);
    // all trees should have the same root
    assert(tree.at(0) == root);

    for (auto const &pair : tree) {
      int id = pair.first * 10000 + pair.second; // current node
      curr_path[pair.second] = id;               // log node in current search

      if (childrens.find(id) == childrens.end()) {
        // init empty set
        childrens[id] = std::set<int>();
      }

      if (pair.second > root_depth) {
        int parent_id = curr_path[pair.second - 1];
        childrens[parent_id].insert(id);
      }
    }
  }

  std::stack<int> q;
  q.push(root_id);

  while (!q.empty()) {
    int curr = q.top();
    q.pop();
    merged_tree.push_back(std::make_pair(curr / 10000, curr % 10000));
    for (int child : childrens[curr]) {
      q.push(child);
    }
  }

  if (verbose) {
    for (auto &pair : merged_tree) {
      std::cout << pair.first << ", depth: " << pair.second << std::endl;
    }
  }

  dfs_tree_inputs[guid] = merged_tree;

  return merged_tree;
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
  assert(request_manager_status == INITIALIZED);
  request_manager_status = SERVING;
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
    rm->serve_incr_decoding(llm);
  } else {
    // Registered SSMs: perform speculative inference
    rm->serve_spec_infer(llm);
  }
}

/*static*/
void RequestManager::serve_incr_decoding(FFModel *llm) {
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
  BatchConfigFuture last_bcf;
  InferenceResultFuture last_irf;
  {
    // Initialize futures for incr decoding
    BatchConfig bc;
    InferenceResult ir;
    last_bcf = Future::from_value<BatchConfig>(bc);
    last_irf = Future::from_value<InferenceResult>(ir);
  }

  std::queue<std::pair<BatchConfigFuture, InferenceResultFuture>>
      batch_pipeline;
  { batch_pipeline.push(std::make_pair(last_bcf, last_irf)); }

  while (!is_background_server_terminated()) {

    if (batch_pipeline.size() >= 4) {
      // Block here to avoid launching too many batches
      auto const &batch = batch_pipeline.front();
      batch.second.get_void_result();
    }
    // deque finished batches
    while (batch_pipeline.size() > 1) {
      auto const &batch = batch_pipeline.front();
      if (batch.second.is_ready()) {
        batch_pipeline.pop();
      } else {
        break;
      }
    }
    runtime->begin_trace(ctx, 12346 /*trace_id*/);
    auto const &next_batch = batch_pipeline.back();
    BatchConfigFuture bcf =
        prepare_next_batch(next_batch.first, next_batch.second, ctx, runtime);
    FutureMap fm = im->inference(llm, 0, bcf);
    assert(fm.get_future_map_domain().get_volume() == 1);
    InferenceResultFuture irf = fm.get_future(0);
    batch_pipeline.push(std::make_pair(bcf, irf));
    last_bcf = bcf;
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
    assert(im->model_weights_loaders.find(llm) !=
           im->model_weights_loaders.end());
    // Load model weights
    im->model_weights_loaders[ssm]->load_weights(ssm);
    // init operators
    im->init_operators_inference(ssm);
  }

  std::queue<std::pair<TreeVerifyBatchConfigFuture, InferenceResultFuture>>
      batch_pipeline;
  // Legion futures for inc_decoding and spec_infer
  TreeVerifyBatchConfigFuture last_tree_bcf;
  InferenceResultFuture last_tree_irf;
  {
    // Initialize futures for spec infer
    TreeVerifyBatchConfig tree_bc;
    InferenceResult tree_ir;
    last_tree_bcf = Future::from_value<TreeVerifyBatchConfig>(tree_bc);
    last_tree_irf = Future::from_value<InferenceResult>(tree_ir);
  }
  batch_pipeline.push(std::make_pair(last_tree_bcf, last_tree_irf));

  while (!is_background_server_terminated()) {

    if (batch_pipeline.size() >= 4) {
      // Block here to avoid launching too many batches
      auto const &batch = batch_pipeline.front();
      batch.second.get_void_result();
    }
    // deque finished batches
    while (batch_pipeline.size() > 1) {
      auto const &batch = batch_pipeline.front();
      if (batch.second.is_ready()) {
        batch_pipeline.pop();
      } else {
        break;
      }
    }
    auto const &next_batch = batch_pipeline.back();
    BeamSearchBatchConfigFuture beam_bcf = prepare_next_batch_init(
        next_batch.first, next_batch.second, 0, ctx, runtime);
    std::vector<BeamSearchBatchConfigFuture> beam_bcf_vec(get_num_ssms());
    for (size_t ssm_id = 0; ssm_id < get_num_ssms(); ssm_id++) {
      beam_bcf_vec[ssm_id] = beam_bcf;
    }
    runtime->begin_trace(ctx, 12345 /*trace_id*/);

    for (size_t i = 0; i < get_num_ssms(); i++) {
      for (int depth = 0; depth < BeamSearchBatchConfig::MAX_BEAM_DEPTH;
           depth++) {
        beam_bcf = beam_bcf_vec[i];

        FutureMap fm = im->inference(get_ssm_model(i), 0, beam_bcf_vec[i]);
        assert(fm.get_future_map_domain().get_volume() == 1);
        BeamInferenceResultFuture beam_irf = fm.get_future(0);
        beam_bcf_vec[i] =
            prepare_next_batch_beam(beam_bcf_vec[i], beam_irf, ctx, runtime);
      }
    }
    // Token Tree Verification
    {
      TreeVerifyBatchConfigFuture tree_bcf =
          prepare_next_batch_verify(beam_bcf_vec, ctx, runtime);
      FutureMap fm = im->inference(llm, 0, tree_bcf);
      assert(fm.get_future_map_domain().get_volume() == 1);
      InferenceResultFuture tree_irf = fm.get_future(0);
      batch_pipeline.push(std::make_pair(tree_bcf, tree_irf));
      last_tree_bcf = tree_bcf;
      last_tree_irf = tree_irf;
    }
    runtime->end_trace(ctx, 12345 /*trace_id*/);
  }
}

void RequestManager::trigger_request_completion_future(
    RequestGuid const &guid) {
  const std::lock_guard<std::mutex> lock(request_to_promise_mutex);
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
  if (request_manager_status == SERVING) {
    request_manager_status = TERMINATED;
    // Wait for the background server to terminate
    Runtime *runtime = Runtime::get_runtime();
    Context ctx = Runtime::get_context();
    background_server_handler.get_void_result();
  }
}

bool RequestManager::is_background_server_terminated() {
  return request_manager_status == TERMINATED;
}

RequestManager *request_manager_singleton = nullptr;

/*static*/
RequestManager *RequestManager::get_request_manager() {
  if (request_manager_singleton == nullptr) {
    request_manager_singleton = new RequestManager();
  }
  return request_manager_singleton;
}

}; // namespace FlexFlow
