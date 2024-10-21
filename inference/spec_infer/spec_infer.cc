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

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string trace_file_path;
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
                      int &max_tokens_per_ssm_batch,
                      int &max_tokens_per_prefilling_batch,
                      int &max_sequence_length,
                      int &max_output_length,
                      int &max_kv_cache_size,
                      int &max_tree_width,
                      int &max_tree_depth,
                      int &expansion_degree,
                      bool &spec_sampling,
                      bool &do_sample,
                      int &sampling_seed,
                      bool &streaming_cache,
                      bool &slo_attainment_early_termination,
                      int &baseline_latency_ms,
                      int &ssm_spec_latency_ms,
                      int &llm_verify_latency_ms,
                      double &request_per_second,
                      bool &spec_infer_old_version,
                      bool &greedy_schedule,
                      bool &equal_schedule,
                      std::string &emission_file_path) {
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
    // traces
    if (!strcmp(argv[i], "-trace")) {
      paths.trace_file_path = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "--max-tokens-per-ssm-batch")) {
      max_tokens_per_ssm_batch = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--max-tokens-per-prefilling-batch")) {
      max_tokens_per_prefilling_batch = std::stoi(argv[++i]);
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
    if (!strcmp(argv[i], "--max-kv-cache-size")) {
      max_kv_cache_size = std::stoi(argv[++i]);
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
    if (!strcmp(argv[i], "--enable-streaming-cache")) {
      streaming_cache = true;
      continue;
    }
    if (!strcmp(argv[i], "--slo-attainment-early-termination")) {
      slo_attainment_early_termination = true;
      continue;
    }
    if (!strcmp(argv[i], "--baseline-latency-ms")) {
      baseline_latency_ms = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--ssm-spec-latency-ms")) {
      ssm_spec_latency_ms = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--llm-verify-latency-ms")) {
      llm_verify_latency_ms = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--request-per-second")) {
      request_per_second = std::stod(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--spec-infer-old-version")) {
      spec_infer_old_version = true;
      continue;
    }
    if (!strcmp(argv[i], "--greedy-schedule")) {
      greedy_schedule = true;
      continue;
    }
    if (!strcmp(argv[i], "--equal-schedule")) {
      equal_schedule = true;
      continue;
    }
    if (!strcmp(argv[i], "--emission-file-path")) {
      emission_file_path = std::string(argv[++i]);
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
  int max_tokens_per_ssm_batch = -1;
  int max_tokens_per_prefilling_batch = -1;
  int max_sequence_length = 512;
  int max_output_length = 512;
  int max_kv_cache_size = -1; // if -1, then use the default value
  int expansion_degree = 3;
  int max_tree_depth = 8;
  int max_tree_width = 16;
  RequestManager::DecodingMode decoding_mode =
      RequestManager::SPECULATIVE_DECODING;
  bool spec_sampling = false;
  bool do_sample = false;
  int sampling_seed = 0;
  bool streaming_cache = false;
  bool slo_attainment_early_termination = false;
  int baseline_latency_ms = 50;
  int ssm_spec_latency_ms = 20;
  int llm_verify_latency_ms = 50;
  double request_per_second = 1.0;
  bool spec_infer_old_version = false;
  bool greedy_schedule = false;
  bool equal_schedule = false;
  std::string emission_file_path;

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
                   max_tokens_per_ssm_batch,
                   max_tokens_per_prefilling_batch,
                   max_sequence_length,
                   max_output_length,
                   max_kv_cache_size,
                   max_tree_width,
                   max_tree_depth,
                   expansion_degree,
                   spec_sampling,
                   do_sample,
                   sampling_seed,
                   streaming_cache,
                   slo_attainment_early_termination,
                   baseline_latency_ms,
                   ssm_spec_latency_ms,
                   llm_verify_latency_ms,
                   request_per_second,
                   spec_infer_old_version,
                   greedy_schedule,
                   equal_schedule,
                   emission_file_path);
  if (max_tokens_per_ssm_batch == -1) {
    max_tokens_per_ssm_batch = max_tokens_per_batch;
  }
  if (max_tokens_per_prefilling_batch == -1) {
    max_tokens_per_prefilling_batch = max_tokens_per_batch;
  }

  get_model_meta(file_paths, model_metadata, use_full_precision);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  // Sanity check for SpecInfer old version
  if (spec_infer_old_version) {
    assert(max_tree_depth = 8);
    assert(max_tree_width >= 3);
    // Total verified tokens
    assert(max_tokens_per_batch >= max_requests_per_batch * 21);
  }

  // Create SentencePiece tokenizer or OPT tokenizer
  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, 0.8, 0.6, spec_sampling, 16);
  InferenceManager *im = InferenceManager::get_inference_manager();
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_ssm_batch(max_tokens_per_ssm_batch);
  rm->set_max_tokens_per_prefilling_batch(max_tokens_per_prefilling_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->set_max_output_length(max_output_length);
  rm->set_max_kv_cache_size(max_kv_cache_size);
  rm->set_max_tree_depth(max_tree_depth);
  rm->set_max_tree_width(max_tree_width);
  rm->set_verbose(verbose);
  rm->set_streaming_cache(streaming_cache);
  rm->register_tokenizer(model_metadata.llm_model_type,
                         model_metadata.bos_token_id,
                         model_metadata.eos_token_id,
                         model_metadata.llm_tokenizer_path);
  rm->set_decoding_mode(decoding_mode);
  rm->set_slo_violation_early_termination(slo_attainment_early_termination);
  rm->set_baseline_latency(baseline_latency_ms);
  rm->set_ssm_spec_latency(ssm_spec_latency_ms);
  rm->set_llm_verify_latency(llm_verify_latency_ms);
  rm->set_spec_infer_old_version(spec_infer_old_version);
  rm->set_greedy_schedule(greedy_schedule);
  rm->set_equal_schedule(equal_schedule);
  rm->register_output_filepath(file_paths.output_file_path);

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
                                streaming_cache,
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

  // Register requests from prompt file
  {
    std::vector<GenerationRequest> requests;
    std::vector<GenerationResult> results;

    if (!file_paths.prompt_file_path.empty()) {
      std::ifstream file_handle(file_paths.prompt_file_path);
      assert(file_handle.good() && "Prompt file does not exist.");
      json prompt_json = json::parse(file_handle,
                                     /*parser_callback_t */ nullptr,
                                     /*allow_exceptions */ true,
                                     /*ignore_comments */ true);
      // Parse slo_ratios
      std::vector<std::pair<double, double>> slo_ratios;
      if (prompt_json[0].contains("slo_ratios")) {
        for (auto &[key, value] : prompt_json[0]["slo_ratios"].items()) {
          slo_ratios.emplace_back(std::stod(key), value.get<double>());
        }
      }
      double total = std::accumulate(
          slo_ratios.begin(),
          slo_ratios.end(),
          0.0,
          [](double sum, std::pair<double, double> const &pair) {
            return sum + pair.second;
          });
      if (std::abs(total - 1.0) > 1e-6) {
        std::cerr << "Error: slo_ratios values do not sum to 1. Total sum: "
                  << total << std::endl;
        assert(false);
      }
      for (size_t i = 1; i < prompt_json.size(); ++i) {
        requests.push_back(GenerationRequest(
            prompt_json[i]["prompt"].get<std::string>(), -1.0, 0));
      }
      PoissonEmissionMachine emission_machine(request_per_second, slo_ratios);
      //   ConstantEmissionMachine emission_machine(-1, slo_ratios);
      results = tree_model.generate(requests, emission_machine);
    } else if (!file_paths.trace_file_path.empty()) {
      std::ifstream file_handle(file_paths.trace_file_path);
      assert(file_handle.good() && "Trace file does not exist.");
      json trace_json = json::parse(file_handle,
                                    /*parser_callback_t */ nullptr,
                                    /*allow_exceptions */ true,
                                    /*ignore_comments */ true);
      std::vector<double> timestamps, ratios;
      for (auto const &json_obj : trace_json) {
        EmissionTrace trace(json_obj);
        requests.push_back(GenerationRequest(trace.prompt, -1.0, 0));
        timestamps.push_back(trace.emission_time_ms);
        ratios.push_back(trace.slo_ratio);
      }
      timestamps.erase(timestamps.begin());
      timestamps.push_back(timestamps.back() + 1000.0);
      TraceEmissionMachine emission_machine(timestamps, ratios);
      results = tree_model.generate(requests, emission_machine);
    } else {
      assert(false && "No prompt or trace file provided.");
    }

    // output generation results as json
    if (!emission_file_path.empty()) {
      json output_json;
      for (size_t i = 0; i < results.size(); ++i) {
        EmissionTrace trace(results[i]);
        output_json.push_back(trace.to_json());
      }
      std::ofstream emission_file_handle(emission_file_path);
      emission_file_handle << output_json.dump(2) << std::endl;
    }
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
