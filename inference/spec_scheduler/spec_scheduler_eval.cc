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
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <wordexp.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <optional>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

LegionRuntime::Logger::Category log_app("llama");

class ConcurrentQueue {
public:
  std::queue<RequestManager::RequestGuid> inf_queue;
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
  }
}

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string warmup_prompt_file_path;
  std::string output_file_path;
};

struct ModelNames {
  std::string llm_model_name;
  std::vector<std::string> ssm_model_names;
};

struct ModelMeta {
  ModelNames model_names;

  ModelType llm_model_type;
  std::string llm_tokenizer_path;
  std::string llm_weights_path;
  std::string llm_model_config_path;

  int bos_token_id, eos_token_id;

  std::vector<ModelType> ssm_model_types;
  std::vector<std::string> ssm_model_config_paths;
  std::vector<std::string> ssm_model_weights_paths;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      ModelNames &model_names,
                      bool &use_full_precision,
                      bool &verbose,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      int &max_spec_tree_token_num,
                      int &max_tree_width,
                      int &max_tree_depth,
                      int &expansion_degree,
                      bool &spec_sampling,
                      bool &do_sample,
                      int &sampling_seed,
                      bool &offline_mode,
                      bool &tpot_slo,
                      double &slo_eps_ms,
                      bool &alignment_test,
                      int &max_buckets_to_run,
                      int &bucket_timeframe) {
  for (int i = 1; i < argc; i++) {
    // llm model name
    if (!strcmp(argv[i], "-llm-model")) {
      model_names.llm_model_name = std::string(argv[++i]);
      for (char &c : model_names.llm_model_name) {
        c = std::tolower(c);
      }
      continue;
    }
    // ssm models names
    if (!strcmp(argv[i], "-ssm-model")) {
      std::string ssm_model_name = std::string(argv[++i]);
      for (char &c : ssm_model_name) {
        c = std::tolower(c);
      }
      model_names.ssm_model_names.push_back(ssm_model_name);
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
    // warmup prompts
    if (!strcmp(argv[i], "-warmup_prompt")) {
      paths.warmup_prompt_file_path = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "--max-spec-tree-token-num")) {
      max_spec_tree_token_num = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tree-width")) {
      max_tree_width = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tree-depth")) {
      max_tree_depth = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--expansion-degree")) {
      expansion_degree = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sampling-seed")) {
      sampling_seed = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--spec-sampling")) {
      spec_sampling = true;
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--offline-mode")) {
      offline_mode = true;
      continue;
    }
    if (!strcmp(argv[i], "--tpot-slo")) {
      tpot_slo = true;
      continue;
    }
    if (!strcmp(argv[i], "--slo-eps")) {
      slo_eps_ms = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--alignment-test")) {
      alignment_test = true;
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

void get_model_meta(FilePaths &file_paths,
                    ModelMeta &model_metadata,
                    bool use_full_precision) {
  if (model_metadata.model_names.llm_model_name.empty() ||
      model_metadata.model_names.ssm_model_names.size() == 0) {
    assert(false && "SpecInfer needs at least one LLM and one SSM for "
                    "speculative inference");
  }
  model_metadata.llm_model_config_path =
      join_path({file_paths.cache_folder_path,
                 "configs",
                 model_metadata.model_names.llm_model_name,
                 "config.json"});
  model_metadata.llm_tokenizer_path =
      join_path({file_paths.cache_folder_path,
                 "tokenizers",
                 model_metadata.model_names.llm_model_name});
  model_metadata.llm_weights_path =
      join_path({file_paths.cache_folder_path,
                 "weights",
                 model_metadata.model_names.llm_model_name,
                 use_full_precision ? "full-precision" : "half-precision"});

  std::ifstream llm_config_file_handle(model_metadata.llm_model_config_path);
  if (!llm_config_file_handle.good()) {
    std::cout << "LLM Model config file "
              << model_metadata.llm_model_config_path << " not found."
              << std::endl;
    assert(false);
  }
  json llm_model_config = json::parse(llm_config_file_handle,
                                      /*parser_callback_t */ nullptr,
                                      /*allow_exceptions */ true,
                                      /*ignore_comments */ true);

  model_metadata.llm_model_type = ModelType::UNKNOWN;
  auto architectures = llm_model_config["architectures"];
  for (auto const &str : architectures) {
    if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM") {
      model_metadata.llm_model_type = ModelType::LLAMA;
      break;
    } else if (str == "OPTForCausalLM") {
      model_metadata.llm_model_type = ModelType::OPT;
      break;
    } else if (str == "RWForCausalLM" || str == "FalconForCausalLM") {
      model_metadata.llm_model_type = ModelType::FALCON;
      break;
    } else if (str == "MPTForCausalLM") {
      model_metadata.llm_model_type = ModelType::MPT;
      break;
    }
  }
  model_metadata.bos_token_id =
      llm_model_config.find("bos_token_id") == llm_model_config.end()
          ? -1
          : (int)llm_model_config.at("bos_token_id");
  model_metadata.eos_token_id =
      llm_model_config.find("eos_token_id") == llm_model_config.end()
          ? -1
          : (int)llm_model_config.at("eos_token_id");

  for (auto ssm_model_name : model_metadata.model_names.ssm_model_names) {
    std::string ssm_config_path = join_path({file_paths.cache_folder_path,
                                             "configs",
                                             ssm_model_name,
                                             "config.json"});
    std::string ssm_tokenizer_path =
        join_path({file_paths.cache_folder_path, "tokenizers", ssm_model_name});
    std::string ssm_weights_path =
        join_path({file_paths.cache_folder_path,
                   "weights",
                   ssm_model_name,
                   use_full_precision ? "full-precision" : "half-precision"});

    std::ifstream ssm_config_file_handle(ssm_config_path);
    if (!ssm_config_file_handle.good()) {
      std::cout << "SSM Model config file " << ssm_config_path << " not found."
                << std::endl;
      assert(false);
    }
    json ssm_model_config = json::parse(ssm_config_file_handle,
                                        /*parser_callback_t */ nullptr,
                                        /*allow_exceptions */ true,
                                        /*ignore_comments */ true);

    ModelType ssm_model_type = ModelType::UNKNOWN;
    auto architectures = ssm_model_config["architectures"];
    for (auto const &str : architectures) {
      if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM") {
        ssm_model_type = ModelType::LLAMA;
        break;
      } else if (str == "OPTForCausalLM") {
        ssm_model_type = ModelType::OPT;
        break;
      } else if (str == "RWForCausalLM") {
        ssm_model_type = ModelType::FALCON;
        break;
      } else if (str == "MPTForCausalLM") {
        ssm_model_type = ModelType::MPT;
        break;
      }
    }
    int ssm_bos_id =
        ssm_model_config.find("bos_token_id") == ssm_model_config.end()
            ? -1
            : (int)ssm_model_config.at("bos_token_id");
    int ssm_eos_id =
        ssm_model_config.find("eos_token_id") == ssm_model_config.end()
            ? -1
            : (int)ssm_model_config.at("eos_token_id");
    if (ssm_bos_id != model_metadata.bos_token_id ||
        ssm_eos_id != model_metadata.eos_token_id) {
      printf("Warning: bos/eos token id mismatch between LLM and one of the "
             "SSMs!\n");
    }
    model_metadata.ssm_model_types.push_back(ssm_model_type);
    model_metadata.ssm_model_config_paths.push_back(ssm_config_path);
    model_metadata.ssm_model_weights_paths.push_back(ssm_weights_path);
  }

  assert(model_metadata.llm_model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");

  for (auto mt : model_metadata.ssm_model_types) {
    if (mt == ModelType::UNKNOWN) {
      assert(false && "One of the SSM model types passed is invalid.");
    }
  }
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  FilePaths file_paths;
  ModelMeta model_metadata;
  bool use_full_precision = false;
  bool verbose = false;
  int max_requests_per_batch = 8;
  int max_tokens_per_batch = 128;
  int max_sequence_length = 512;
  int max_spec_tree_token_num = 64;
  int expansion_degree = 3;
  int max_tree_depth = 8;
  int max_tree_width = 16;
  RequestManager::DecodingMode decoding_mode =
      RequestManager::SPECULATIVE_DECODING;
  bool spec_sampling = false;
  bool do_sample = false;
  bool offline_mode = false;
  bool tpot_slo = false;
  double slo_eps_ms = 0.0;
  bool alignment_test = false;
  int sampling_seed = 0;
  int max_buckets_to_run = 100000;
  int bucket_timespan = 1;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   model_metadata.model_names,
                   use_full_precision,
                   verbose,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   max_spec_tree_token_num,
                   max_tree_width,
                   max_tree_depth,
                   expansion_degree,
                   spec_sampling,
                   do_sample,
                   sampling_seed,
                   offline_mode,
                   tpot_slo,
                   slo_eps_ms,
                   alignment_test,
                   max_buckets_to_run,
                   bucket_timespan);

  get_model_meta(file_paths, model_metadata, use_full_precision);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  // Create SentencePiece tokenizer or OPT tokenizer
  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, 0.8, 0.6, spec_sampling, 16);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_spec_tree_token_num(max_spec_tree_token_num);
  rm->set_max_sequence_length(max_sequence_length);
  rm->set_max_tree_depth(max_tree_depth);
  rm->set_max_tree_width(max_tree_width);
  rm->set_verbose(verbose);
  rm->register_tokenizer(model_metadata.llm_model_type,
                         model_metadata.bos_token_id,
                         model_metadata.eos_token_id,
                         model_metadata.llm_tokenizer_path);
  rm->set_decoding_mode(decoding_mode);
  rm->register_output_filepath(file_paths.output_file_path);
  rm->use_tpot_slo(tpot_slo);
  rm->set_slo_eps_ms(slo_eps_ms);
  rm->set_alignment_test(alignment_test);

  // Create LLM model
  FFModel tree_model(ffconfig, ffconfig.cpu_offload);
  if (model_metadata.llm_model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(tree_model,
                              model_metadata.llm_model_config_path,
                              model_metadata.llm_weights_path,
                              TREE_VERIFY_MODE,
                              generationConfig,
                              use_full_precision);
  } else if (model_metadata.llm_model_type == ModelType::OPT) {
    OPT::create_opt_model(tree_model,
                          model_metadata.llm_model_config_path,
                          model_metadata.llm_weights_path,
                          TREE_VERIFY_MODE,
                          use_full_precision);
  } else if (model_metadata.llm_model_type == ModelType::FALCON) {
    FALCON::create_falcon_model(tree_model,
                                model_metadata.llm_model_config_path,
                                model_metadata.llm_weights_path,
                                TREE_VERIFY_MODE,
                                use_full_precision);
  } else if (model_metadata.llm_model_type == ModelType::MPT) {
    MPT::create_mpt_model(tree_model,
                          model_metadata.llm_model_config_path,
                          model_metadata.llm_weights_path,
                          TREE_VERIFY_MODE,
                          generationConfig,
                          use_full_precision);
  } else {
    assert(false && "Invalid LLM model type passed (or no type was passed).");
  }

  // Create SSM models
  int num_ssms = model_metadata.ssm_model_types.size();
  std::vector<int> ssm_model_ids;
  std::vector<FFModel> ssm_models;
  FFConfig bm_config = ffconfig;
  bm_config.data_parallelism_degree = bm_config.tensor_parallelism_degree =
      bm_config.pipeline_parallelism_degree = 1;
  //   bm_config.data_parallelism_degree = 1;
  //   bm_config.tensor_parallelism_degree = 4;
  //   bm_config.pipeline_parallelism_degree = 1;
  for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
    FFModel beam_model(bm_config);
    ssm_models.push_back(beam_model);
  }

  for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
    FFModel &beam_model = ssm_models[ssm_id];
    if (model_metadata.ssm_model_types[ssm_id] == ModelType::LLAMA) {
      LLAMA::create_llama_model(beam_model,
                                model_metadata.ssm_model_config_paths[ssm_id],
                                model_metadata.ssm_model_weights_paths[ssm_id],
                                TREE_SEARCH_MODE,
                                generationConfig,
                                use_full_precision);
    } else if (model_metadata.ssm_model_types[ssm_id] == ModelType::OPT) {
      OPT::create_opt_model(beam_model,
                            model_metadata.ssm_model_config_paths[ssm_id],
                            model_metadata.ssm_model_weights_paths[ssm_id],
                            TREE_SEARCH_MODE,
                            use_full_precision);
    } else if (model_metadata.ssm_model_types[ssm_id] == ModelType::FALCON) {
      FALCON::create_falcon_model(
          beam_model,
          model_metadata.ssm_model_config_paths[ssm_id],
          model_metadata.ssm_model_weights_paths[ssm_id],
          TREE_SEARCH_MODE,
          use_full_precision);
    } else if (model_metadata.ssm_model_types[ssm_id] == ModelType::MPT) {
      MPT::create_mpt_model(beam_model,
                            model_metadata.ssm_model_config_paths[ssm_id],
                            model_metadata.ssm_model_weights_paths[ssm_id],
                            TREE_SEARCH_MODE,
                            generationConfig,
                            use_full_precision);
    } else {
      assert(false && "Invalid SSM model type passed.");
    }

    rm->register_ssm_model(&beam_model);
  }

  rm->start_background_server(&tree_model);

  // Warmup stage
  std::cout << "======= WARMUP STAGE =======" << std::endl;
  {
    using json = nlohmann::json;
    std::ifstream file_handle(file_paths.warmup_prompt_file_path);
    assert(file_handle.good() && "Prompt file does not exist.");
    json prompt_json = json::parse(file_handle,
                                   /*parser_callback_t */ nullptr,
                                   /*allow_exceptions */ true,
                                   /*ignore_comments */ true);

    std::vector<std::pair<std::string, std::optional<double>>> prompts;
    // The json should be a list of elements, where an element is 
    // either a string, or a tuple [string, float]. 
    // The string is the prompt, and the float is an optional tpot SLO.
    for (auto &prompt : prompt_json) {
      if (prompt.is_string()) { // The element doesn't contain an SLO
        std::string text = prompt.get<std::string>();
        prompts.emplace_back(text, std::nullopt);
      } else { // The element contains an SLO
        std::string text = prompt[0].get<std::string>();
        double tpot_slo_ms = prompt[1].get<double>();
        prompts.emplace_back(text, tpot_slo_ms);
      }
    }
    tree_model.generate(prompts, -1 /*not used in this implem*/);
  }
  std::cout << "===== END WARMUP STAGE =====" << std::endl;
  rm->reset_profiling_stats();

  // Now run workload!

  // Load all requests in advance
  nb_millisecs = nb_millisecs * bucket_timespan;
  using json = nlohmann::json;
  std::ifstream file_handle(file_paths.prompt_file_path);
  assert(file_handle.good() && "Prompt file does not exist.");
  json prompt_json = json::parse(file_handle,
                                /*parser_callback_t */ nullptr,
                                /*allow_exceptions */ true,
                                /*ignore_comments */ true);

  auto const &lists = prompt_json.get<std::vector<std::vector<json>>>();
  std::vector<size_t> bucket_arrival_times_ms;
  std::vector<std::vector<std::pair<std::string, std::optional<double>>>> buckets;

  size_t index = 0;
  size_t nb_prompts = 0;
  size_t prompt_limit = 10;
  for (auto const &list : lists) {
    if (nb_prompts >= prompt_limit) {
      break;
    }
    if (!list.empty()) {
      bucket_arrival_times_ms.push_back(nb_millisecs * index);
      std::vector<std::pair<std::string, std::optional<double>>> prompts;
      for (auto const prompt : list) {
        if (nb_prompts >= prompt_limit) {
          break;
        }
        if (prompt.is_string()) { // The element doesn't contain an SLO
          prompts.emplace_back(prompt.get<std::string>(), std::nullopt);
        } else { // The element contains an SLO
          prompts.emplace_back(prompt[0].get<std::string>(), prompt[1].get<double>());
        }
        nb_prompts++;
      }
      buckets.push_back(prompts);
    }
    index++;
  }
  assert(bucket_arrival_times_ms.size() == buckets.size() &&
        "Bucket arrival times and buckets are not the same size");

  if (offline_mode) {

    std::vector<std::pair<std::string, std::optional<double>>> grouped_prompts;
    // The json should be a list of elements, where an element is 
    // either a string, or a tuple [string, float]. 
    // The string is the prompt, and the float is an optional tpot SLO.
    for (auto &bckt : buckets) {
      for (auto &prompt : bckt) {
        grouped_prompts.emplace_back(prompt.first, prompt.second);
      }
    }
    tree_model.generate(grouped_prompts, -1 /*not used in this implem*/);

  } else {

    ConcurrentQueue *guids = get_common_guids_queue();
    std::thread consumer{consume};
    {

      // Replay the trace of inference requests
      auto start_time = std::chrono::steady_clock::now();
      for (int i = 0; i < bucket_arrival_times_ms.size(); i++) {
        if (bucket_arrival_times_ms[i] >= max_buckets_to_run * nb_millisecs) {
          break;
        }
        // sleep until bucket arrives
        auto bucket_arrival_time =
            start_time +
            std::chrono::milliseconds(bucket_arrival_times_ms[i]);
        std::this_thread::sleep_until(bucket_arrival_time);

        // create inference requests for the bucket
        {
          const std::lock_guard<std::mutex> lock(guids->request_queue_mutex);
          for (auto const prompt : buckets[i]) {
            RequestManager::RequestGuid guid = RequestManager::INVALID_GUID;
            if (prompt.second) { // Contains SLO
              guid = rm->register_new_request(prompt.first, *prompt.second);
            } else {
              guid = rm->register_new_request(prompt.first);
            }
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

  }

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}
