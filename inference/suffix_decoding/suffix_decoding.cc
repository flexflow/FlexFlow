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

#include "suffix_decoding/utils.h"

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

Legion::Logger log_app("llama");


void process_partition(RequestManager *rm, std::string input_filename) {
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffconfig;
  FilePaths file_paths;
  ModelMeta model_metadata;
  std::string partition_name;
  bool use_full_precision = false;
  bool verbose = false;
  int max_requests_per_batch = 16;
  int max_tokens_per_batch = 256;
  int max_sequence_length = 1024;
  int max_spec_tree_token_num = 23;
  int expansion_degree = 1;

  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  parse_input_args(argv,
                   argc,
                   file_paths,
                   model_metadata.model_names,
                   partition_name,
                   use_full_precision,
                   verbose,
                   max_requests_per_batch,
                   max_tokens_per_batch,
                   max_sequence_length,
                   expansion_degree);

  get_model_meta(file_paths, model_metadata, use_full_precision);

  assert(ffconfig.data_parallelism_degree * ffconfig.tensor_parallelism_degree *
             ffconfig.pipeline_parallelism_degree ==
         ffconfig.numNodes * ffconfig.workersPerNode);
  
  json trace = load_trace(file_paths.prompt_file_path);
  json training_entries = get_training_entries(trace, partition_name);
  json eval_entries = get_eval_entries(trace, partition_name);

  GenerationConfig generationConfig;
  InferenceManager *im = InferenceManager::get_inference_manager();
  RequestManager *rm = RequestManager::get_request_manager();
  init_request_manager(rm,
                       model_metadata,
                       file_paths,
                       max_requests_per_batch,
                       max_tokens_per_batch,
                       max_spec_tree_token_num,
                       max_sequence_length,
                       expansion_degree);

  // Create LLM model
  FFModel tree_model(ffconfig, ffconfig.cpu_offload);
  init_llm(tree_model, model_metadata, generationConfig, use_full_precision);

  // Create SSM models
  int num_ssms = model_metadata.ssm_model_types.size();
  std::vector<FFModel> ssm_models;
  FFConfig bm_config = ffconfig;
  bm_config.data_parallelism_degree = bm_config.tensor_parallelism_degree =
      bm_config.pipeline_parallelism_degree = 1;
  for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
    FFModel beam_model(bm_config);
    ssm_models.push_back(beam_model);
  }
  init_ssms(rm, ssm_models, num_ssms, model_metadata, generationConfig, use_full_precision);
  
  rm->start_background_server(&tree_model);

  int total_num_requests = 0;
  {
    std::vector<Request> requests;
    for (auto entry: eval_entries) {
      std::string prompt = entry["prompt"];
      int response_length = entry["response_length"];
      // printf("Prompt[%d]: %s\n", total_num_requests, prompt.c_str());
      // Add inference request
      Request inference_req;
      inference_req.prompt = prompt;
      inference_req.max_new_tokens = response_length;
      requests.push_back(inference_req);
      total_num_requests++;
    }
    tree_model.generate(requests);
  }  

  // Register requests from prompt file
  // int total_num_requests = 0;
  // {
  //   using json = nlohmann::json;
  //   std::ifstream file_handle(file_paths.prompt_file_path);
  //   assert(file_handle.good() && "Prompt file does not exist.");
  //   json prompt_json = json::parse(file_handle,
  //                                  /*parser_callback_t */ nullptr,
  //                                  /*allow_exceptions */ true,
  //                                  /*ignore_comments */ true);

  //   std::vector<Request> requests;
  //   for (auto &prompt : prompt_json) {
  //     std::string text = prompt.get<std::string>();
  //     printf("Prompt[%d]: %s\n", total_num_requests, text.c_str());
  //     // Add inference request
  //     Request inference_req;
  //     inference_req.prompt = text;
  //     inference_req.max_length = 128;
  //     requests.push_back(inference_req);
  //     total_num_requests++;
  //   }
  //   tree_model.generate(requests);
  // }

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
