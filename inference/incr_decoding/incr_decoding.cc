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
#include "models/opt.h"

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

void parse_input_args(char **argv,
                      int argc,
                      FilePaths &paths,
                      ModelType &llm_model_type,
                      bool &use_full_precision,
                      bool &verbose,
                      bool &do_sample,
                      float &temperature,
                      float &topp,
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
        llm_model_type = ModelType::LLAMA;
      } else if (model_type_str == "opt") {
        llm_model_type = ModelType::OPT;
      } else if (model_type_str == "falcon") {
        llm_model_type = ModelType::FALCON;
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
  }
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
  ModelType model_type;
  bool use_full_precision = false;
  bool verbose = false;
  bool do_sample = false;
  float temperature = 0.0f;
  float topp = 0.0f;
  size_t num_devices = ffconfig.workersPerNode * ffconfig.numNodes;
  int data_parallelism_degree = 1, tensor_parallelism_degree = 1,
      pipeline_parallelism_degree = 1;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   model_type,
                   use_full_precision,
                   verbose,
                   do_sample,
                   temperature,
                   topp,
                   data_parallelism_degree,
                   tensor_parallelism_degree,
                   pipeline_parallelism_degree);
  ffconfig.data_parallelism_degree = data_parallelism_degree;
  ffconfig.tensor_parallelism_degree = tensor_parallelism_degree;
  ffconfig.pipeline_parallelism_degree = pipeline_parallelism_degree;

  assert(data_parallelism_degree * tensor_parallelism_degree *
             pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);

  assert(model_type != ModelType::UNKNOWN &&
         "Invalid LLM model type passed (or no type was passed).");

  SamplingConfig samplingConfig(do_sample, temperature, topp);
  InferenceManager im(ffconfig, BatchConfig::MAX_NUM_TOKENS);
  RequestManager rm(model_type,
                    file_paths.tokenizer_file_path,
                    /*verbose*/ verbose,
                    file_paths.output_file_path);

  FFModel model(ffconfig, ffconfig.cpu_offload);
  if (model_type == ModelType::LLAMA) {
    LLAMA::create_llama_model(model,
                              im,
                              file_paths.llm_config_file_path,
                              file_paths.llm_weight_file_path,
                              INC_DECODING_MODE,
                              samplingConfig,
                              use_full_precision);
  } else if (model_type == ModelType::OPT) {
    OPT::create_opt_model(model,
                          im,
                          file_paths.llm_config_file_path,
                          file_paths.llm_weight_file_path,
                          INC_DECODING_MODE,
                          use_full_precision);
  } else if (model_type == ModelType::FALCON) {
    FALCON::create_falcon_model(model,
                                im,
                                file_paths.llm_config_file_path,
                                file_paths.llm_weight_file_path,
                                INC_DECODING_MODE,
                                use_full_precision);
  } else {
    assert(false && "unknow model type");
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
    bc = rm.prepare_next_batch(bc, ir);
    if (rm.get_num_processed_requests() >= total_num_requests) {
      break;
    }
    FutureMap fm = im.inference(&model, 0, bc);
    assert(fm.get_future_map_domain().get_volume() == 1);
    Future future = fm.get_future(0);
    ir = future.get_result<InferenceResult>();
    // assert(false);
  }

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
