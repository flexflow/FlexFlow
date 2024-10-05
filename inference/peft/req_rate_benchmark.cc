/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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
#include "flexflow/request_manager.h"
#include "inference/models/falcon.h"
#include "inference/models/llama.h"
#include "inference/models/mpt.h"
#include "inference/models/opt.h"
#include "inference/models/starcoder.h"
#include <chrono>
#include <mutex>
#include <thread>
#include <wordexp.h>

#include <nlohmann/json.hpp>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");

class ConcurrentQueue {
public:
  std::queue<RequestManager::RequestGuid> inf_queue;
  std::queue<RequestManager::RequestGuid> peft_queue;
  std::mutex request_queue_mutex;
  bool producer_finished = false;
};

ConcurrentQueue *common_guids_singleton = nullptr;
int nb_millisecs = 1000; // Default bucket timeframe is 1 second

ConcurrentQueue *get_common_guids_queue() {
  if (common_guids_singleton == nullptr) {
    common_guids_singleton = new ConcurrentQueue();
  }
  return common_guids_singleton;
}

void consume() {
  RequestManager *rm = RequestManager::get_request_manager();
  ConcurrentQueue *guids = get_common_guids_queue();
  bool producer_is_finished = false;
  bool queue_is_empty = false;
  // int i=0;
  while (!producer_is_finished || !queue_is_empty) {
    RequestManager::RequestGuid guid = RequestManager::INVALID_GUID;
    {
      const std::lock_guard<std::mutex> lock(guids->request_queue_mutex);
      queue_is_empty = guids->inf_queue.empty();
      producer_is_finished = guids->producer_finished;
      if (!queue_is_empty) {
        guid = guids->inf_queue.front();
        guids->inf_queue.pop();
      }
    }
    if (guid != RequestManager::INVALID_GUID) {
      GenerationResult result = rm->get_generation_result(guid);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(nb_millisecs));
    }
    // i++;
    // cout << "Iteration " << i;
  }
  rm->set_inference_finished();

  while (guids->peft_queue.size() > 0) {
    GenerationResult result =
        rm->get_generation_result(guids->peft_queue.front());
    guids->peft_queue.pop();
  }
}

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string output_file_path;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      std::string &peft_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &do_sample,
                      bool &enable_peft,
                      float &temperature,
                      float &topp,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      int &max_buckets_to_run,
                      int &bucket_timeframe) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    if (!strcmp(argv[i], "-enable-peft")) {
      enable_peft = true;
      continue;
    }
    if (!strcmp(argv[i], "-peft-model")) {
      peft_model_name = std::string(argv[++i]);
      for (char &c : peft_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // cache folder
    if (!strcmp(argv[i], "-cache-folder")) {
      paths.cache_folder_path = std::string(argv[++i]);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-output-file")) {
      paths.output_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--use-full-precision")) {
      use_full_precision = true;
      continue;
    }
    // verbose logging to stdout
    if (!strcmp(argv[i], "--verbose")) {
      verbose = true;
      continue;
    }
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--temperature")) {
      temperature = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--topp")) {
      topp = std::stof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-requests-per-batch")) {
      max_requests_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tokens-per-batch")) {
      max_tokens_per_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-sequence-length")) {
      max_sequence_length = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-buckets-to-run")) {
      max_buckets_to_run = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--bucket-timeframe")) {
      bucket_timeframe = std::stoi(argv[++i]);
      continue;
    }
  }
  if (paths.cache_folder_path.empty()) {
    char const *ff_cache_path = std::getenv("FF_CACHE_PATH");
    paths.cache_folder_path = ff_cache_path ? std::string(ff_cache_path)
                                            : std::string("~/.cache/flexflow");
  }
  // Expand ~ to the home directory if needed
  wordexp_t p;
  wordexp(paths.cache_folder_path.c_str(), &p, 0);
  paths.cache_folder_path = p.we_wordv[0];
  wordfree(&p);
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  if (ffconfig.cpu_offload == false && ffconfig.quantization_type != DT_NONE) {
    assert(false && "Doesn't support quantization in non-offload mode");
  }
  FilePaths file_paths;
  std::string llm_model_name, peft_model_name;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  bool enable_peft = false;
  float temperature = 0.0f;
  float topp = 0.0f;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 128;
  int max_sequence_length = 256;
  int max_buckets_to_run = 1000000000;
  bool enable_peft_finetuning = false;
  int bucket_timespan = 1;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   peft_model_name,
                   use_full_precision,
                   verbose,
                   do_sample,
                   enable_peft,
                   temperature,
                   topp,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   max_buckets_to_run,
                   bucket_timespan);
  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  std::string config_filepath = join_path(
      {file_paths.cache_folder_path, "configs", llm_model_name, "config.json"});
  std::string tokenizer_filepath =
      join_path({file_paths.cache_folder_path, "tokenizers", llm_model_name});
  std::string weights_filepath =
      join_path({file_paths.cache_folder_path,
                 "weights",
                 llm_model_name,
                 use_full_precision ? "full-precision" : "half-precision"});
  std::ifstream config_file_handle(config_filepath);
  if (!config_file_handle.good()) {
    std::cout << "Model config file " << config_filepath << " not found."
              << std::endl;
    assert(false);
  }
  if (enable_peft && peft_model_name.empty()) {
    std::cout << "PEFT enabled, but no PEFT model id passed" << std::endl;
    assert(false);
  } else if (!enable_peft && !peft_model_name.empty()) {
    std::cout << "PEFT model id passed, but PEFT is not enabled" << std::endl;
    assert(false);
  }

  json model_config = json::parse(config_file_handle,
                                  /*parser_callback_t */ nullptr,
                                  /*allow_exceptions */ true,
                                  /*ignore_comments */ true);
  ModelType model_type = ModelType::UNKNOWN;
  auto architectures = model_config["architectures"];
  for (auto const &str : architectures) {
    if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM") {
      model_type = ModelType::LLAMA;
      break;
    } else if (str == "OPTForCausalLM") {
      model_type = ModelType::OPT;
      break;
    } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
      model_type = ModelType::FALCON;
      break;
    } else if (str == "GPTBigCodeForCausalLM") {
      model_type = ModelType::STARCODER;
      break;
    } else if (str == "MPTForCausalLM") {
      model_type = ModelType::MPT;
      break;
    }
  }
  int bos_token_id = model_config.find("bos_token_id") == model_config.end()
                         ? -1
                         : (int)model_config.at("bos_token_id");
  int eos_token_id = model_config.find("eos_token_id") == model_config.end()
                         ? -1
                         : (int)model_config.at("eos_token_id");

  assert(model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");

  // load PEFT config
  LoraLinearConfig peft_config =
      peft_model_name.empty()
          ? LoraLinearConfig::EmptyConfig
          : LoraLinearConfig(file_paths.cache_folder_path, peft_model_name);

  GenerationConfig generationConfig(do_sample, temperature, topp);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(
      max_requests_per_batch +
      (int)enable_peft_finetuning); // add one slot for finetuning if needed
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->register_tokenizer(
      model_type, bos_token_id, eos_token_id, tokenizer_filepath);
  rm->register_output_filepath(file_paths.output_file_path);
  rm->set_enable_peft_finetuning(enable_peft_finetuning);

  FFModel model(ffconfig, ffconfig.cpu_offload);
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              config_filepath,
                              weights_filepath,
                              INC_DECODING_MODE,
                              generationConfig,
                              use_full_precision);
  } else if (model_type == ModelType::OPT) {
    OPT::create_opt_model(model,
                          config_filepath,
                          weights_filepath,
                          INC_DECODING_MODE,
                          use_full_precision);
  } else if (model_type == ModelType::FALCON) {
    FALCON::create_falcon_model(model,
                                config_filepath,
                                weights_filepath,
                                INC_DECODING_MODE,
                                use_full_precision);
  } else if (model_type == ModelType::STARCODER) {
    STARCODER::create_starcoder_model(model,
                                      config_filepath,
                                      weights_filepath,
                                      INC_DECODING_MODE,
                                      generationConfig,
                                      use_full_precision);
  } else if (model_type == ModelType::MPT) {
    MPT::create_mpt_model(model,
                          config_filepath,
                          weights_filepath,
                          INC_DECODING_MODE,
                          generationConfig,
                          use_full_precision);
  } else {
    assert(false && "unknow model type");
  }

  // Add PEFT layer
  PEFTModelID *peft_model_id = nullptr;
  if (!peft_model_name.empty()) {
    peft_model_id = model.add_lora_layer(peft_config);
  }

  rm->start_background_server(&model);

  // Warmup stage
  {
    std::vector<Request> requests;
    for (int i = 0; i < 100; i++) {
      Request inference_req;
      inference_req.benchmarking_tokens = 128;
      inference_req.max_length = 256;
      inference_req.warmup = true;
      inference_req.peft_model_id =
          (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
      requests.push_back(inference_req);
    }

    Request fine_tuning_req;
    fine_tuning_req.req_type = RequestType::REQ_FINETUNING;
    fine_tuning_req.benchmarking_tokens = 1024;
    fine_tuning_req.max_length = 1024;
    fine_tuning_req.warmup = true;
    fine_tuning_req.peft_model_id =
        (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
    fine_tuning_req.max_training_steps = 1;
    requests.push_back(fine_tuning_req);
    std::vector<GenerationResult> result = model.generate(requests);
  }

  rm->set_inference_finished(false); // reset inference finished flag
  std::cout << "----------warmup finished--------------" << std::endl;

  // Now run online workload!

  nb_millisecs = nb_millisecs * bucket_timespan;
  int total_num_requests = 0;
  int num_arrival_buckets = 0;
  ConcurrentQueue *guids = get_common_guids_queue();
  std::thread consumer{consume};
  {

    // Load all requests in advance
    using json = nlohmann::json;
    std::ifstream file_handle(file_paths.prompt_file_path);
    assert(file_handle.good() && "Prompt file does not exist.");
    json prompt_json = json::parse(file_handle,
                                   /*parser_callback_t */ nullptr,
                                   /*allow_exceptions */ true,
                                   /*ignore_comments */ true);

    auto const &lists = prompt_json.get<std::vector<std::vector<json>>>();
    std::vector<size_t> bucket_arrival_times_s;
    std::vector<std::vector<std::pair<int, int>>> buckets;

    size_t index = 0;
    for (auto const &list : lists) {
      if (!list.empty()) {
        bucket_arrival_times_s.push_back(index);
        std::vector<std::pair<int, int>> prompts;
        for (auto const &dict : list) {
          int prompt_length = dict["human"];
          int sequence_length = dict["gpt"];
          assert(prompt_length + sequence_length <= max_sequence_length &&
                 "Prompt + sequence length exceeds max sequence length");
          prompts.push_back(std::make_pair(prompt_length, sequence_length));
        }
        buckets.push_back(prompts);
      }
      index++;
    }
    assert(bucket_arrival_times_s.size() == buckets.size() &&
           "Bucket arrival times and buckets are not the same size");
    // for (int i=0; i<10; i++) {
    //   printf("bucket_arrival_times_s[%i]: %i\n", i,
    //   bucket_arrival_times_s[i]); printf("bucket[%i]: %i\n", i,
    //   buckets[i].size()); for (const auto& prompt : buckets[i]) {
    //     printf("\tprompt: %i, %i\n", prompt.first, prompt.second);
    //   }
    // }

    // Add fine-tuning request
    Request fine_tuning_req;
    fine_tuning_req.req_type = RequestType::REQ_FINETUNING;
    fine_tuning_req.benchmarking_tokens = 1024;
    fine_tuning_req.max_length = 1024;
    fine_tuning_req.peft_model_id =
        (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
    fine_tuning_req.max_training_steps = 1000000000;
    RequestManager::RequestGuid ft_guid =
        rm->register_new_peft_request(fine_tuning_req);
    if (ft_guid != RequestManager::INVALID_GUID) {
      const std::lock_guard<std::mutex> lock(guids->request_queue_mutex);
      guids->peft_queue.push(ft_guid);
    }

    // Replay the trace of inference requests
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < bucket_arrival_times_s.size(); i++) {
      if (bucket_arrival_times_s[i] >= max_buckets_to_run) {
        break;
      }
      // sleep until bucket arrives
      auto bucket_arrival_time =
          start_time +
          std::chrono::milliseconds(bucket_arrival_times_s[i] * nb_millisecs);
      std::this_thread::sleep_until(bucket_arrival_time);

      // create inference requests for the bucket
      std::vector<Request> requests;
      for (auto const &prompt : buckets[i]) {
        // printf("Prompt length: %d, sequence length: %d\n", prompt_length,
        // sequence_length);
        Request inference_req;
        inference_req.benchmarking_tokens = prompt.first;
        inference_req.max_length = prompt.second + prompt.first;
        inference_req.peft_model_id =
            (peft_model_id != nullptr) ? *peft_model_id : PEFTModelID::NO_ID;
        requests.push_back(inference_req);
      }

      {
        const std::lock_guard<std::mutex> lock(guids->request_queue_mutex);
        for (int i = 0; i < requests.size(); i++) {
          RequestManager::RequestGuid guid =
              rm->register_new_request(requests.at(i));
          if (guid != RequestManager::INVALID_GUID) {
            guids->inf_queue.push(guid);
          }
        }
      }
    }

    { // Notify the consumer that no more requests are incoming
      const std::lock_guard<std::mutex> lock(guids->request_queue_mutex);
      guids->producer_finished = true;
    }
  }

  // Wait for consumer to finish
  consumer.join();

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;

  // free tokenizer space in memory
}

void FlexFlow::register_custom_tasks() {}
