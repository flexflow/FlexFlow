#include "flexflow/inference.h"
#include "models/falcon.h"
#include "models/llama.h"
#include "models/mpt.h"
#include "models/opt.h"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <wordexp.h>

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

struct FilePaths {
  std::string cache_folder_path;
  std::string prompt_file_path;
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
                      std::string &partition_name,
                      bool &use_full_precision,
                      bool &verbose,
                      int &max_requests_per_batch,
                      int &max_tokens_per_batch,
                      int &max_sequence_length,
                      int &expansion_degree);

void get_model_meta(FilePaths &file_paths,
                    ModelMeta &model_metadata,
                    bool use_full_precision);

void init_request_manager(RequestManager *rm, ModelMeta &model_metadata,
                          FilePaths &file_paths, int max_requests_per_batch,
                          int max_tokens_per_batch, int max_spec_tree_token_num,
                          int max_sequence_length, int expansion_degree);

void init_llm(FFModel &tree_model, ModelMeta &model_metadata, 
                GenerationConfig &generationConfig, bool use_full_precision);

void init_ssms(RequestManager *rm, std::vector<FFModel> &ssm_models, int num_ssms,
                ModelMeta &model_metadata, GenerationConfig &generationConfig,
                bool use_full_precision);

json load_trace(std::string filename);
json get_training_entries(json data, std::string partition_name);
json get_eval_entries(json data, std::string partition_name);