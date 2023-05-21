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
  std::vector<std::string> ssm_weight_file_paths;
  std::vector<std::string> ssm_config_file_paths;
  std::string prompt_file_path;
  std::string tokenizer_file_path;
};

enum ModelType { UNKNOWN, LLAMA, OPT };

struct ModelTypes {
  ModelType llm_model_type;
  std::vector<ModelType> ssm_model_types;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      ModelTypes &model_types) {
  for (int i = 1; i < argc; i++) {
    // llm model type
    if (!strcmp(argv[i], "-llm-model")) {
      std::string model_type_str = std::string(argv[++i]);
      std::transform(model_type_str.begin(),
                     model_type_str.end(),
                     model_type_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (model_type_str == "llama") {
        model_types.llm_model_type = ModelType::LLAMA;
      } else if (model_type_str == "opt") {
        model_types.llm_model_type = ModelType::OPT;
      } else {
        model_types.llm_model_type = ModelType::UNKNOWN;
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
    // ssm models types
    if (!strcmp(argv[i], "-ssm-model")) {
      std::string model_type_str = std::string(argv[++i]);
      std::transform(model_type_str.begin(),
                     model_type_str.end(),
                     model_type_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (model_type_str == "llama") {
        model_types.ssm_model_types.push_back(ModelType::LLAMA);
      } else if (model_type_str == "opt") {
        model_types.ssm_model_types.push_back(ModelType::OPT);
      } else {
        model_types.ssm_model_types.push_back(ModelType::UNKNOWN);
      }
      continue;
    }
    // ssm model weights
    if (!strcmp(argv[i], "-ssm-weight")) {
      std::string file_path = std::string(argv[++i]);
      paths.ssm_weight_file_paths.push_back(file_path);
      continue;
    }
    // ssm model configs
    if (!strcmp(argv[i], "-ssm-config")) {
      std::string file_path = std::string(argv[++i]);
      paths.ssm_config_file_paths.push_back(file_path);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
      continue;
    }
    // tokenizer
    if (!strcmp(argv[i], "-tokenizer")) {
      std::string token_folder = std::string(argv[++i]);
      if (!token_folder.empty() && token_folder.back() != '/')
        paths.tokenizer_file_path = token_folder + '/';
      else
        paths.tokenizer_file_path = token_folder;
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
  ModelTypes model_types;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv, argc, file_paths, model_types);
  if (file_paths.ssm_weight_file_paths.size() == 0) {
    assert(false &&
           "SpecInfer needs at least one SSM for speculative inference");
  }
  if (file_paths.ssm_config_file_paths.size() !=
      file_paths.ssm_weight_file_paths.size()) {
    assert(false && "Number of SSM config files passed does not match number "
                    "of SSM weights");
  }
  assert(model_types.llm_model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");
  if (model_types.ssm_model_types.size() !=
      file_paths.ssm_weight_file_paths.size()) {
    assert(false && "Number of valid SSM model types passed does not match "
                    "number of SSM weights");
  }
  for (auto mt : model_types.ssm_model_types) {
    if (mt == ModelType::UNKNOWN) {
      assert(false && "One of the SSM model types passed is invalid.");
    }
  }

  // Create SentencePiece tokenizer or OPT tokenizer
  std::unique_ptr<Tokenizer> tokenizer;
  if (model_types.llm_model_type == ModelType::LLAMA) {
    tokenizer = std::make_unique<SentencePieceTokenizer>(file_paths.tokenizer_file_path);
  } else {
    // check that the path leads to a folder
    std::string vocab_file = file_paths.tokenizer_file_path + "gpt2-vocab.json";
    std::string merges_file = file_paths.tokenizer_file_path + "gpt2-merges.txt";
    std::filesystem::path path1(vocab_file);
    std::filesystem::path path2(merges_file);
    assert(std::filesystem::exists(path1) && "Vocab file gpt2-vocab.json does not exist at the specified path");
    assert(std::filesystem::exists(path2) && "Merge file gpt2-merges.txt does not exist at the specified path");
    tokenizer = std::make_unique<OptTokenizer>(vocab_file, merges_file);
  }

  InferenceManager im(ffconfig, BatchConfig::MAX_NUM_TOKENS, 1);
  RequestManager rm(tokenizer.get());
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

  FFModel beam_model(ffconfig);
  FFModel tree_model(ffconfig);
  if (model_types.ssm_model_types[0] == ModelType::LLAMA) {
    LLAMA::create_llama_model(beam_model,
                              im,
                              file_paths.ssm_config_file_paths[0],
                              file_paths.ssm_weight_file_paths[0],
                              1,
                              BEAM_SEARCH_MODE);
  } else {
    OPT::create_opt_model(beam_model,
                          im,
                          file_paths.ssm_config_file_paths[0],
                          file_paths.ssm_weight_file_paths[0],
                          1,
                          BEAM_SEARCH_MODE);
  }
  if (model_types.llm_model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(tree_model,
                              im,
                              file_paths.llm_config_file_path,
                              file_paths.llm_weight_file_path,
                              ffconfig.workersPerNode * ffconfig.numNodes,
                              TREE_VERIFY_MODE);
  } else {
    OPT::create_opt_model(tree_model,
                          im,
                          file_paths.llm_config_file_path,
                          file_paths.llm_weight_file_path,
                          ffconfig.workersPerNode * ffconfig.numNodes,
                          TREE_VERIFY_MODE);
  }

  TreeVerifyBatchConfig tree_bc;
  BeamSearchBatchConfig beam_bc;
  InferenceResult tree_ir;

  while (rm.get_num_processed_requests() < total_num_requests) {
    int depth = 0;
    // Beam Search
    beam_bc = rm.prepare_next_batch_init(tree_bc, tree_ir);
    if (rm.get_num_processed_requests() >= total_num_requests) {
      break;
    }
    while (true) {
      depth = beam_bc.beamRequestsInfo[0].current_depth;
      FutureMap fm = im.inference(&beam_model, 0, beam_bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      Future future = fm.get_future(0);
      BeamInferenceResult beam_ir = future.get_result<BeamInferenceResult>();
      if (depth - 1 >= BeamSearchBatchConfig::MAX_BEAM_DEPTH) {
        break;
      } else {
        beam_bc = rm.prepare_next_batch_beam(beam_bc, beam_ir);
      }
    }
    // Token Tree Verification
    {
      tree_bc = rm.prepare_next_batch_verify(beam_bc);
      FutureMap fm = im.inference(&tree_model, 0, tree_bc);
      assert(fm.get_future_map_domain().get_volume() == 1);
      Future future = fm.get_future(0);
      tree_ir = future.get_result<InferenceResult>();
    }
  }

  // Execution fence
  {
    Future future = runtime->issue_execution_fence(ctx);
    future.get_void_result();
  }

  // float* data
  std::cout << "----------inference finished--------------" << std::endl;
}

void FlexFlow::register_custom_tasks() {}
