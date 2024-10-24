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
#include <cassert>
#include <filesystem>
#include <string>
#include <wordexp.h>
#include "flexflow/request_manager.h"

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;
using RequestGuid = BatchConfig::RequestGuid;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string trace_file_path;
  std::string trace_output_path;
  std::string log_file_path;
  std::string csv_file_path;
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
                      int &max_output_length,
                      int &max_tree_width,
                      int &max_tree_depth,
                      int &expansion_degree,
                      bool &do_sample,
                      double &request_per_second,
                      bool &add_special_tokens,
                      std::string &target_partition) {
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
    // trace
    if (!strcmp(argv[i], "-trace")) {
      paths.trace_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-trace-output-path")) {
      paths.trace_output_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-target-partition")) {
      target_partition = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-log-output-path")) {
      paths.log_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-csv-output-path")) {
      paths.csv_file_path = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "--max-output-length")) {
      max_output_length = std::stoi(argv[++i]);
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
    if (!strcmp(argv[i], "--do-sample")) {
      do_sample = true;
      continue;
    }
    if (!strcmp(argv[i], "--request-per-second")) {
      request_per_second = std::stod(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--add-special-tokens")) {
      add_special_tokens = true;
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
    if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM" || str == "MistralForCausalLM") {
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
      if (str == "LlamaForCausalLM" || str == "LLaMAForCausalLM" || str == "MistralForCausalLM") {
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
  int max_output_length = 512;
  int expansion_degree = 3;
  int max_tree_depth = 8;
  int max_tree_width = 16;
  RequestManager::DecodingMode decoding_mode =
      RequestManager::SPECULATIVE_DECODING;
  bool do_sample = false;
  int sampling_seed = 0;
  double request_per_second = 1.0;
  bool add_special_tokens = false;
  std::string target_partition = "FEATURE_EXTRACTION";

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
                   max_output_length,
                   max_tree_width,
                   max_tree_depth,
                   expansion_degree,
                   do_sample,
                   request_per_second,
                   add_special_tokens,
                   target_partition);

  get_model_meta(file_paths, model_metadata, use_full_precision);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  // using json = nlohmann::json;
  using json = nlohmann::ordered_json;
  std::ifstream input_file(file_paths.trace_file_path);
  assert(input_file.good() && "Prompt file does not exist.");
  json j;
  input_file >> j;
  input_file.close();

  // Find the partition with name "FEATURE_EXTRACTION"
  auto &partitions = j["partitions"];
  auto it =
      std::find_if(partitions.begin(),
                   partitions.end(),
                   [target_partition](json const &partition) {
                     return partition["partition_name"] == target_partition;
                   });
  json &partition = *it;
  if (it == partitions.end()) {
    std::cerr << "Partition " << target_partition
              << " not found in the trace file." << std::endl;
    assert(false);
  }
  // check that the max prompt + response length sum in the eval_entries in the
  // partition does not exceed the max_sequence_length
  int max_prompt_response_length = 0;
  for (auto &eval_entry : partition["eval_entries"]) {
    int prompt_length = eval_entry["prompt_length"];
    int response_length = eval_entry["response_length"];
    if (response_length >= max_output_length) {
      std::cerr << "Error: A response length from the targt partition in the "
                   "dataset (="
                << response_length
                << ") exceeds the max_output_length(=" << max_output_length
                << ")." << std::endl;
      assert(false);
    }
    max_prompt_response_length =
        std::max(max_prompt_response_length, prompt_length + response_length);
  }
  if (max_prompt_response_length >= max_sequence_length) {
    std::cerr << "Error: max prompt + response length sum (="
              << max_prompt_response_length
              << ") in the eval_entries in the partition exceeds the "
                 "max_sequence_length(="
              << max_sequence_length << ")." << std::endl;
    assert(false);
  }

  // Sanity check for SpecInfer old version
  assert(max_tree_depth <= 8);
  assert(max_tree_width >= 3);
  // Total verified tokens
  assert(max_tokens_per_batch >= max_requests_per_batch * 21);

  // Create SentencePiece tokenizer or OPT tokenizer
  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, 0.8, 0.6, false, 16);
  InferenceManager *im = InferenceManager::get_inference_manager();
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_ssm_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_prefilling_batch(max_tokens_per_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->set_max_output_length(max_output_length);
  rm->set_max_tree_depth(max_tree_depth);
  rm->set_max_tree_width(max_tree_width);
  rm->set_verbose(verbose);
  rm->set_streaming_cache(false);
  rm->register_tokenizer(model_metadata.llm_model_type,
                         model_metadata.bos_token_id,
                         model_metadata.eos_token_id,
                         model_metadata.llm_tokenizer_path);
  rm->set_decoding_mode(decoding_mode);
  rm->set_slo_violation_early_termination(false);
  rm->set_baseline_latency(50);
  rm->set_ssm_spec_latency(20);
  rm->set_llm_verify_latency(50);
  rm->set_spec_infer_old_version(true);
  rm->set_greedy_schedule(false);
  rm->set_equal_schedule(false);
  rm->register_output_filepath(file_paths.log_file_path);

  // Create LLM model
  FFModel tree_model(ffconfig, ffconfig.cpu_offload);
  if (model_metadata.llm_model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(tree_model,
                              model_metadata.llm_model_config_path,
                              model_metadata.llm_weights_path,
                              TREE_VERIFY_MODE,
                              generationConfig,
                              false,
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
                                false,
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

  int total_num_requests = 0;
  {
    // Iterate through eval_entries
    std::vector<GenerationRequest> requests;
    std::vector<double> timestamps, ratios;
    for (auto &entry : partition["eval_entries"]) {
      std::string text = entry["prompt"];
      int max_new_tokens_ = entry["response_length"];
      // printf("Prompt[%d]: %s\n", total_num_requests, text.c_str());
      GenerationRequest inference_req(text, -1.0, 0, add_special_tokens);
      // inference_req.prompt = text;
      // inference_req.slo_ratio = -1.0;
      // inference_req.emission_time_ms = 0;
      // // inference_req.max_new_tokens = max_new_tokens_;
      // inference_req.add_special_tokens = false;
      requests.push_back(inference_req);
      timestamps.push_back(0);
      ratios.push_back(1.0);
      total_num_requests++;
      // break;
    }
    TraceEmissionMachine emission_machine(timestamps, ratios);
    std::vector<GenerationResult> result =
        tree_model.generate(requests, emission_machine);
    assert(result.size() == requests.size());
    assert(result.size() == total_num_requests);
    assert(result.size() == partition["eval_entries"].size());
    int i = 0;
    for (auto &entry : partition["eval_entries"]) {
      entry["original_response"] = entry["response"];
      entry["original_response_length"] = entry["response_length"];
      std::string ff_out = result[i].output_text;
      int tot_length = result[i].output_text.length();
      entry["response"] = ff_out;
      entry["response_length"] = result[i].output_tokens.size();
      entry["specinfer_decoding_steps"] = result[i].decoding_steps;
      i++;
    }

    // Write the modified JSON to a file
    std::ofstream output_file(file_paths.trace_output_path);
    if (output_file.is_open()) {
      output_file << j.dump(2);
      output_file.close();
      std::cout << "Modified JSON has been saved to "
                << file_paths.trace_output_path << std::endl;
    } else {
      std::cerr << "Unable to open file for writing." << std::endl;
    }
  }

  // terminate the request manager by stopping the background thread
  rm->terminate_background_server();

  // get profliling results
  std::unordered_map<RequestGuid, RequestProfileInfo> profiling_results = rm->get_requests_profiling();
  std::unordered_map<RequestGuid, GenerationResult> request_generation_results = rm->get_request_generation_results();
  // save profiling results to csv file
  std::string header = "llm,ssm,batch_size,tokens_per_batch,mean_decoding_steps,mean_output_length,mean_e2e_latency,mean_llm_ttft,mean_llm_tpot,mean_ssm_step_time,mean_candidate_size";
  std::string row = "";
  row += model_metadata.model_names.llm_model_name + ",";
  // first ssm
  assert(model_metadata.model_names.ssm_model_names.size() == 1);
  row += model_metadata.model_names.ssm_model_names[0] + ",";
  row += std::to_string(max_requests_per_batch) + ",";
  row += std::to_string(max_tokens_per_batch) + ",";
  double mean_decoding_steps = 0;
  double mean_output_length = 0;
  double mean_e2e_latency = 0;
  double mean_llm_ttft = 0;
  double mean_llm_tpot = 0;
  double mean_ssm_step_time = 0;
  double mean_candidate_size = 0;

  for (auto &profiling_result : profiling_results) {
    RequestGuid guid = profiling_result.first;
    RequestProfileInfo &profile_info = profiling_result.second;
    GenerationResult &result = request_generation_results[guid];
    mean_decoding_steps += profile_info.llm_decoding_steps;
    mean_output_length += result.output_tokens.size();
    mean_e2e_latency += profile_info.finish_time - profile_info.start_time;
    // LLM ttft
    double prefilling_time_ms = 0.0;
    if (profile_info.start_decoding_time != 0) {
      prefilling_time_ms = (profile_info.start_decoding_time - profile_info.start_time) / 1000.0;
    } else {
      prefilling_time_ms = (profile_info.finish_time - profile_info.start_time) / 1000.0;
    }
    mean_llm_ttft += prefilling_time_ms;
    // LLM tpot
    double per_token_time_ms = 0;
    if (profile_info.start_decoding_time != 0) {
      per_token_time_ms = (profile_info.finish_time - profile_info.start_decoding_time) / 1000.0 / result.output_tokens.size();
    }
    mean_llm_tpot += per_token_time_ms;
  }
  mean_decoding_steps /= profiling_results.size();
  mean_output_length /= profiling_results.size();
  mean_e2e_latency /= profiling_results.size();
  mean_llm_ttft /= profiling_results.size();
  mean_llm_tpot /= profiling_results.size();
  row += std::to_string(mean_decoding_steps) + ",";
  row += std::to_string(mean_output_length) + ",";
  row += std::to_string(mean_e2e_latency) + ",";
  row += std::to_string(mean_llm_ttft) + ",";
  row += std::to_string(mean_llm_tpot) + ",";
  
  ProfileInfo profile_info = rm->get_profiling_info();
  // SSM tpots
  for (double time : profile_info.ssm_step_times) {
    mean_ssm_step_time += time;
  }
  mean_ssm_step_time /= profile_info.ssm_step_times.size();
  // SSM number of steps (= candidate length)
  for (int nb : profile_info.ssm_steps) {
    mean_candidate_size += nb;
  }
  mean_candidate_size /= profile_info.ssm_steps.size();
  row += std::to_string(mean_ssm_step_time) + ",";
  row += std::to_string(mean_candidate_size);

  // csv filepath
  // create csv filepath and add header if it doesn't exist
  bool csv_file_exists = std::filesystem::exists(file_paths.csv_file_path);
  if (!csv_file_exists) {
    // Create new file and write header
    std::ofstream file(file_paths.csv_file_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << file_paths.csv_file_path << std::endl;
      assert(false);
    }
    file << header << "\n";
    file.close();
  }
  
  // Append the new row
  std::ofstream file(file_paths.csv_file_path, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_paths.csv_file_path << std::endl;
  }
  file << row << "\n";
  file.close();

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}
