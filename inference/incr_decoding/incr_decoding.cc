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
#include "flexflow/tokenizers.h"
#include "models/llama.h"
#include "models/opt.h"
#include <filesystem>
#include <nlohmann/json.hpp>

using namespace Legion;

LegionRuntime::Logger::Category log_app("llama");

struct FilePaths {
  std::string llm_weight_file_path;
  std::string llm_config_file_path;
  std::string prompt_file_path;
  std::string tokenizer_file_path;
  std::string output_file_path;
};

enum ModelType { UNKNOWN, LLAMA, OPT };

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      ModelType &llm_model_type,
                      bool &use_full_precision,
                      bool &verbose) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      std::string model_type_str = std::string(argv[++i]);
      std::transform(model_type_str.begin(),
                     model_type_str.end(),
                     model_type_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (model_type_str == "llama") {
        llm_model_type = ModelType::LLAMA;
      } else if (model_type_str == "opt") {
        llm_model_type = ModelType::OPT;
      } else {
        llm_model_type = ModelType::UNKNOWN;
      }
      continue;
    }
    // llm model weights
    if (!strcmp(argv[i], "-llm-weight")) {
      paths.llm_weight_file_path = std::string(argv[++i]);
      continue;
    }
    // llm model configs
    if (!strcmp(argv[i], "-llm-config")) {
      paths.llm_config_file_path = std::string(argv[++i]);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
      continue;
    }
    // tokenizer
    if (!strcmp(argv[i], "-tokenizer")) {
      paths.tokenizer_file_path = std::string(argv[++i]);
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
  }
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  FilePaths file_paths;
  ModelType model_type;
  bool use_full_precision = false;
  bool verbose = false;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(
      argv, argc, file_paths, model_type, use_full_precision, verbose);

  assert(model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");

  // Create SentencePiece tokenizer or OPT tokenizer
  SentencePieceTokenizer *sp_tokenizer = nullptr;
  OptTokenizer *opt_tokenizer = nullptr;
  if (model_type == ModelType::LLAMA) {
    sp_tokenizer = new SentencePieceTokenizer(file_paths.tokenizer_file_path);
  } else {
    std::string tokenizer_folder =
        (!file_paths.tokenizer_file_path.empty() &&
         file_paths.tokenizer_file_path.back() != '/')
            ? file_paths.tokenizer_file_path + '/'
            : file_paths.tokenizer_file_path;
    std::string vocab_file = tokenizer_folder + "gpt2-vocab.json";
    std::string merges_file = tokenizer_folder + "gpt2-merges.txt";
    std::filesystem::path path1(vocab_file);
    std::filesystem::path path2(merges_file);
    assert(std::filesystem::exists(path1) &&
           "Vocab file gpt2-vocab.json does not exist at the specified path");
    assert(std::filesystem::exists(path2) &&
           "Merge file gpt2-merges.txt does not exist at the specified path");
    opt_tokenizer = new OptTokenizer(vocab_file, merges_file);
  }

  InferenceManager im(ffconfig, BatchConfig::MAX_NUM_TOKENS, 1);
  RequestManager rm((model_type == ModelType::LLAMA)
                        ? (Tokenizer *)sp_tokenizer
                        : (Tokenizer *)opt_tokenizer,
                    /*verbose*/ verbose,
                    file_paths.output_file_path);

  FFModel model(ffconfig);
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              im,
                              file_paths.llm_config_file_path,
                              file_paths.llm_weight_file_path,
                              ffconfig.workersPerNode * ffconfig.numNodes,
                              INC_DECODING_MODE,
                              use_full_precision);
  } else {
    assert(model_type == ModelType::OPT);
    OPT::create_opt_model(model,
                          im,
                          file_paths.llm_config_file_path,
                          file_paths.llm_weight_file_path,
                          ffconfig.workersPerNode * ffconfig.numNodes,
                          INC_DECODING_MODE,
                          use_full_precision);
  }

  int total_num_requests = 0;
  {
    using json = nlohmann::json;
    std::ifstream file_handle(file_paths.prompt_file_path);
    assert(file_handle.good() && "Prompt file does not exist.");
    json prompt_json = json::parse(file_handle,
                                   /*parser_callback_t */ nullptr,
                                   /*allow_exceptions */ true,
                                   /*ignore_comments */ true);
    for (auto &prompt : prompt_json) {
      std::string text = prompt.get<std::string>();
      printf("Prompt[%d]: %s\n", total_num_requests, text.c_str());
      total_num_requests++;
      rm.register_new_request(text, 128 /*max_sequence_length*/);
    }
  }

  BatchConfig bc;
  InferenceResult ir;
  while (rm.get_num_processed_requests() < total_num_requests) {
    bc = ffconfig.cpu_offload ? rm.prepare_next_batch_offload(bc, ir)
                              : rm.prepare_next_batch(bc, ir);
    if (rm.get_num_processed_requests() >= total_num_requests) {
      break;
    }
    FutureMap fm = im.inference(&model, 0, bc);
    assert(fm.get_future_map_domain().get_volume() == 1);
    Future future = fm.get_future(0);
    ir = future.get_result<InferenceResult>();
  }

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;

  // free tokenizer space in memory
  if (model_type == ModelType::LLAMA) {
    delete sp_tokenizer;
  } else {
    delete opt_tokenizer;
  }
}

void FlexFlow::register_custom_tasks() {}
