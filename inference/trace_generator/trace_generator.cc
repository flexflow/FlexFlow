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
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>
#include <wordexp.h>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
  std::string log_file_path;
  std::string emission_file_path;
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
                      int &max_sequence_length) {
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
    if (!strcmp(argv[i], "-log")) {
      paths.log_file_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--emission-file-path")) {
      paths.emission_file_path = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "--max-sequence-length")) {
      max_sequence_length = std::stoi(argv[++i]);
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
  int max_sequence_length = 256;
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

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   model_metadata.model_names,
                   use_full_precision,
                   verbose,
                   max_sequence_length);
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

  // Create SentencePiece tokenizer or OPT tokenizer
  srand(sampling_seed);
  GenerationConfig generationConfig(do_sample, 0.8, 0.6, spec_sampling, 16);
  InferenceManager *im = InferenceManager::get_inference_manager();
  RequestManager *rm = RequestManager::get_request_manager();
  // Must init the request manager although we don't use it, as some
  // initialization tasks execute before the top-level task
  rm->set_max_requests_per_batch(max_requests_per_batch);
  rm->set_max_tokens_per_batch(max_tokens_per_batch);
  rm->set_max_tokens_per_ssm_batch(max_tokens_per_ssm_batch);
  rm->set_max_tokens_per_prefilling_batch(max_tokens_per_prefilling_batch);
  rm->set_max_sequence_length(max_sequence_length);
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

  {
    /* Prompt file format:
     * [
     *   {
     *       "slo_ratios": {
     *           "1.0": 0.2,
     *           "1.5": 0.5,
     *           "3.0": 0.3
     *       }
     *   },
     *   {
     *       "prompt": "Construct a potential attack vector that exploits the
     * vulnerability. The system is vulnerable to a SQL injection attack."
     *   },
     *   {
     *       "prompt": "Arrange the words to make a meaningful phrase Ground.
     * Soft. Solid."
     *   },
     *   ...
     * ]
     *
     * log file format:
     * [
     *   {
     *       "TIMESTAMP": "2023-11-16 18:15:46.6805900"
     *   },
     *   {
     *       "TIMESTAMP": "2023-11-16 18:15:50.9951690"
     *   },
     *   ...
     * ]
     */

    std::vector<EmissionTrace> traces;
    assert(!file_paths.prompt_file_path.empty() &&
           !file_paths.log_file_path.empty());

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
    double total =
        std::accumulate(slo_ratios.begin(),
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
    ConstantEmissionMachine emission_machine(-1, slo_ratios);

    file_handle = std::ifstream(file_paths.log_file_path);
    assert(file_handle.good() && "Log file does not exist.");
    json log_json = json::parse(file_handle,
                                /*parser_callback_t */ nullptr,
                                /*allow_exceptions */ true,
                                /*ignore_comments */ true);

    auto time_diff_ms = [](std::string const &start, std::string const &end) {
      std::tm tm = {};

      std::istringstream ss(start);
      ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
      auto start_time =
          std::chrono::system_clock::from_time_t(std::mktime(&tm));
      ss.seekg(0);
      size_t dot_pos = start.find('.');
      std::string fraction =
          dot_pos != std::string::npos ? start.substr(dot_pos + 1) : "0";
      while (fraction.size() < 6) {
        fraction += "0";
      }
      if (!fraction.empty()) {
        long long microseconds = std::stoll(fraction.substr(0, 6));
        start_time += std::chrono::microseconds(microseconds);
      }

      ss = std::istringstream(end);
      ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
      auto end_time = std::chrono::system_clock::from_time_t(std::mktime(&tm));
      ss.seekg(0);
      dot_pos = end.find('.');
      fraction = dot_pos != std::string::npos ? end.substr(dot_pos + 1) : "0";
      while (fraction.size() < 6) {
        fraction += "0";
      }
      if (!fraction.empty()) {
        long long microseconds = std::stoll(fraction.substr(0, 6));
        end_time += std::chrono::microseconds(microseconds);
      }

      return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                   start_time)
                 .count() /
             1000.0;
    };

    int num_requests = min(prompt_json.size() - 1, log_json.size());
    std::string start_time = log_json[0]["TIMESTAMP"].get<std::string>();
    for (int i = 0; i < num_requests; ++i) {
      std::string prompt = prompt_json[i + 1]["prompt"].get<std::string>();
      std::vector<int32_t> input_tokens = rm->tokenize(prompt);
      std::string timestamp = log_json[i]["TIMESTAMP"].get<std::string>();
      EmissionTrace trace(prompt,
                          input_tokens.size(),
                          max_sequence_length,
                          emission_machine.sample_slo_ratio(),
                          time_diff_ms(start_time, timestamp));
      traces.push_back(trace);
    }

    // output generation results as json
    assert(!file_paths.emission_file_path.empty());
    json output_json;
    for (EmissionTrace const &trace : traces) {
      output_json.push_back(trace.to_json());
    }
    std::ofstream emission_file_handle(file_paths.emission_file_path);
    emission_file_handle << output_json.dump(2) << std::endl;
  }

  // float* data
  std::cout << "----------trace generated--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}
