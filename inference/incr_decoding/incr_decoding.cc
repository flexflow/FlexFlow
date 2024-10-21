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
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include "models/starcoder.h"
#include <cassert>
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

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      std::string &llm_model_name,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &do_sample,
                      float &temperature,
                      float &topp,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_tokens_per_ssm_batch,
                      int &max_tokens_per_prefilling_batch,
                      int &max_sequence_length,
                      int &max_output_length,
                      int &max_kv_cache_size,
                      int &sampling_seed,
                      bool &streaming_cache,
                      bool &slo_attainment_early_termination,
                      int &baseline_latency_ms,
                      int &ssm_spec_latency_ms,
                      int &llm_verify_latency_ms,
                      double &request_per_second,
                      std::string &emission_file_path) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      llm_model_name = std::string(argv[++i]);
      for (char &c : llm_model_name) {
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
    if (!strcmp(argv[i], "--sampling-seed")) {
      sampling_seed = std::stoi(argv[++i]);
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

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  if (ffconfig.cpu_offload == false && ffconfig.quantization_type != DT_NONE) {
    assert(false && "Doesn't support quantization in non-offload mode");
  }
  FilePaths file_paths;
  std::string llm_model_name;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  float temperature = 0.8f;
  float topp = 0.6f;
  int max_requests_per_batch = 1;
  int max_tokens_per_batch = 128;
  int max_tokens_per_ssm_batch = -1;
  int max_tokens_per_prefilling_batch = -1;
  int max_sequence_length = 256;
  int max_output_length = 512;
  int max_kv_cache_size = -1; //if -1, then use the default value
  RequestManager::DecodingMode decoding_mode =
      RequestManager::INCREMENTAL_DECODING;
  int sampling_seed = 0;
  bool streaming_cache = false;
  bool slo_attainment_early_termination = false;
  int baseline_latency_ms = 50;
  int ssm_spec_latency_ms = 20;
  int llm_verify_latency_ms = 50;
  double request_per_second = 1.0;
  std::string emission_file_path;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   llm_model_name,
                   use_full_precision,
                   verbose,
                   do_sample,
                   temperature,
                   topp,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_tokens_per_ssm_batch,
                   max_tokens_per_prefilling_batch,
                   max_sequence_length,
                   max_output_length,
                   max_kv_cache_size,
                   sampling_seed,
                   streaming_cache,
                   slo_attainment_early_termination,
                   baseline_latency_ms,
                   ssm_spec_latency_ms,
                   llm_verify_latency_ms,
                   request_per_second,
                   emission_file_path);
  if (max_tokens_per_ssm_batch == -1) {
    max_tokens_per_ssm_batch = max_tokens_per_batch;
  }
  if (max_tokens_per_prefilling_batch == -1) {
    max_tokens_per_prefilling_batch = max_tokens_per_batch;
  }

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

  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, temperature, topp);
  RequestManager *rm = RequestManager::get_request_manager();
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_ssm_batch(max_tokens_per_ssm_batch);
  rm->set_max_tokens_per_prefilling_batch(max_tokens_per_prefilling_batch);
  rm->set_max_sequence_length(max_sequence_length);
  rm->set_max_output_length(max_output_length);
  rm->set_max_kv_cache_size(max_kv_cache_size);
  rm->set_decoding_mode(decoding_mode);
  rm->set_slo_violation_early_termination(slo_attainment_early_termination);
  rm->set_baseline_latency(baseline_latency_ms);
  rm->set_ssm_spec_latency(ssm_spec_latency_ms);
  rm->set_llm_verify_latency(llm_verify_latency_ms);
  rm->set_max_tree_depth(8);
  rm->set_max_tree_width(16);
  rm->set_verbose(verbose);
  rm->set_streaming_cache(streaming_cache);
  rm->register_tokenizer(
      model_type, bos_token_id, eos_token_id, tokenizer_filepath);
  rm->register_output_filepath(file_paths.output_file_path);

  FFModel model(ffconfig, ffconfig.cpu_offload);
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              config_filepath,
                              weights_filepath,
                              INC_DECODING_MODE,
                              generationConfig,
                              streaming_cache,
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

  rm->start_background_server(&model);

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
      // ConstantEmissionMachine emission_machine(-1, slo_ratios);
      results = model.generate(requests, emission_machine);
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
      TraceEmissionMachine emission_machine(timestamps, ratios);
      results = model.generate(requests, emission_machine);
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

  // free tokenizer space in memory
}

void FlexFlow::register_custom_tasks() {}
