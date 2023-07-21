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
  std::string output_file_path;
};

struct ModelTypes {
  ModelType llm_model_type;
  std::vector<ModelType> ssm_model_types;
};

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      ModelTypes &model_types,
                      bool &use_full_precision,
                      bool &verbose,
                      int &data_parallelism_degree,
                      int &tensor_parallelism_degree,
                      int &pipeline_parallelism_degree) {
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
      paths.tokenizer_file_path = std::string(argv[++i]);
      continue;
    }
    // output file
    if (!strcmp(argv[i], "-output-file")) {
      paths.output_file_path = std::string(argv[++i]);
      continue;
    }
    // data parallelism degree
    if (!strcmp(argv[i], "-data-parallelism-degree")) {
      data_parallelism_degree = std::stoi(argv[++i]);
      continue;
    }
    // tensor parallelism degree
    if (!strcmp(argv[i], "-tensor-parallelism-degree")) {
      tensor_parallelism_degree = std::stoi(argv[++i]);
      continue;
    }
    // pipeline parallelism degree
    if (!strcmp(argv[i], "-pipeline-parallelism-degree")) {
      pipeline_parallelism_degree = std::stoi(argv[++i]);
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
  ModelTypes model_types;
  bool use_full_precision = false;
  bool verbose = false;
  size_t num_devices = ffconfig.workersPerNode * ffconfig.numNodes;
  int data_parallelism_degree = 1, tensor_parallelism_degree = 1,
      pipeline_parallelism_degree = 1;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   model_types,
                   use_full_precision,
                   verbose,
                   data_parallelism_degree,
                   tensor_parallelism_degree,
                   pipeline_parallelism_degree);
  ffconfig.data_parallelism_degree = data_parallelism_degree;
  ffconfig.tensor_parallelism_degree = tensor_parallelism_degree;
  ffconfig.pipeline_parallelism_degree = pipeline_parallelism_degree;
  assert(data_parallelism_degree * tensor_parallelism_degree *
             pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

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
  SamplingConfig samplingConfig;
  InferenceManager im(ffconfig, BatchConfig::MAX_NUM_TOKENS);
  RequestManager rm(model_types.llm_model_type,
                    file_paths.tokenizer_file_path,
                    /*verbose*/ verbose,
                    file_paths.output_file_path);

  // Create LLM model
  FFModel tree_model(ffconfig, ffconfig.cpu_offload);
  if (model_types.llm_model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(tree_model,
                              im,
                              file_paths.llm_config_file_path,
                              file_paths.llm_weight_file_path,
                              TREE_VERIFY_MODE,
                              samplingConfig,
                              use_full_precision);
  } else if (model_types.llm_model_type == ModelType::OPT) {
    OPT::create_opt_model(tree_model,
                          im,
                          file_paths.llm_config_file_path,
                          file_paths.llm_weight_file_path,
                          TREE_VERIFY_MODE,
                          use_full_precision);
  } else {
    assert(false && "Invalid LLM model type passed (or no type was passed).");
  }

  // Create SSM models
  int num_ssms = model_types.ssm_model_types.size();
  std::vector<int> ssm_model_ids;
  std::vector<FFModel> ssm_models;
  FFConfig bm_config = ffconfig;
  bm_config.data_parallelism_degree = bm_config.tensor_parallelism_degree =
      bm_config.pipeline_parallelism_degree = 1;
  for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
    FFModel beam_model(bm_config);
    ssm_models.push_back(beam_model);
  }

  for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
    FFModel &beam_model = ssm_models[ssm_id];
    if (model_types.ssm_model_types[ssm_id] == ModelType::LLAMA) {
      LLAMA::create_llama_model(beam_model,
                                im,
                                file_paths.ssm_config_file_paths[ssm_id],
                                file_paths.ssm_weight_file_paths[ssm_id],
                                BEAM_SEARCH_MODE,
                                samplingConfig,
                                use_full_precision);
    } else if (model_types.ssm_model_types[ssm_id] == ModelType::OPT) {
      OPT::create_opt_model(beam_model,
                            im,
                            file_paths.ssm_config_file_paths[ssm_id],
                            file_paths.ssm_weight_file_paths[ssm_id],
                            BEAM_SEARCH_MODE,
                            use_full_precision);
    } else {
      assert(false && "Invalid SSM model type passed.");
    }

    int beam_model_id = rm.register_new_model(&beam_model);
    ssm_model_ids.push_back(beam_model_id);
  }

  // Register requests from prompt file
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

  TreeVerifyBatchConfigFuture tree_bcf;
  BeamSearchBatchConfigFuture beam_bcf;
  InferenceResultFuture tree_irf;
  std::vector<BeamSearchBatchConfigFuture> beam_bcf_vec;
  {
    TreeVerifyBatchConfig tree_bc;
    BeamSearchBatchConfig beam_bc;
    InferenceResult tree_ir;
    tree_bcf = Future::from_value<TreeVerifyBatchConfig>(tree_bc);
    beam_bcf = Future::from_value<BeamSearchBatchConfig>(beam_bc);
    tree_irf = Future::from_value<InferenceResult>(tree_ir);
    for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
      beam_bcf_vec.push_back(Future::from_value<BeamSearchBatchConfig>(
          BeamSearchBatchConfig(ssm_model_ids[ssm_id])));
    }
  }

  while (rm.get_num_processed_requests() < total_num_requests) {
    // Beam Search
    beam_bcf = rm.prepare_next_batch_init(tree_bcf, tree_irf, 0);
    for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
      beam_bcf_vec[ssm_id] = beam_bcf;
    }

    if (rm.get_num_processed_requests() >= total_num_requests) {
      break;
    }

    for (int i = 0; i < num_ssms; i++) {
      for (int depth = 0; depth < BeamSearchBatchConfig::MAX_BEAM_DEPTH;
           depth++) {
        beam_bcf = beam_bcf_vec[i];

        FutureMap fm = im.inference(rm.get_model(0), 0, beam_bcf_vec[i]);
        assert(fm.get_future_map_domain().get_volume() == 1);
        BeamInferenceResultFuture beam_irf = fm.get_future(0);
        beam_bcf_vec[i] = rm.prepare_next_batch_beam(beam_bcf_vec[i], beam_irf);
      }
      // std::cout << "----------beam search finished for model "
      //           << beam_bc_vec[i].model_id << "------------" << std::endl;
    }
    // Token Tree Verification
    {
      tree_bcf = rm.prepare_next_batch_verify(beam_bcf_vec);
      FutureMap fm = im.inference(&tree_model, 0, tree_bcf);
      assert(fm.get_future_map_domain().get_volume() == 1);
      tree_irf = fm.get_future(0);
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
