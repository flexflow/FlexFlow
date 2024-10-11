#include "suffix_decoding/utils.h"

using namespace FlexFlow;
using namespace Legion;
using json = nlohmann::json;

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
                      int &expansion_degree) {
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
    // cache folder
    if (!strcmp(argv[i], "-partition-name")) {
      partition_name = std::string(argv[++i]);
      continue;
    }
    // prompts
    if (!strcmp(argv[i], "-prompt")) {
      paths.prompt_file_path = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "--max-sequence-length")) {
      max_sequence_length = std::stoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--expansion-degree")) {
      expansion_degree = std::stoi(argv[++i]);
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

void init_request_manager(RequestManager *rm, ModelMeta &model_metadata,
                          FilePaths &file_paths, int max_requests_per_batch,
                          int max_tokens_per_batch, int max_spec_tree_token_num, int max_sequence_length,
                          int expansion_degree) {
    rm->set_max_requests_per_batch(max_requests_per_batch);
    rm->set_max_tokens_per_batch(max_tokens_per_batch);
    rm->set_max_spec_tree_token_num(max_spec_tree_token_num);
    rm->set_max_sequence_length(max_sequence_length);
    rm->register_tokenizer(model_metadata.llm_model_type,
                            model_metadata.bos_token_id,
                            model_metadata.eos_token_id,
                            model_metadata.llm_tokenizer_path);
    rm->register_output_filepath(file_paths.output_file_path);

    // first decoding step: 3 results
    if (expansion_degree != -1) {
        rm->push_spec_infer_tree_width(1);
        rm->push_spec_infer_tree_width(1);
        rm->push_spec_infer_tree_width(expansion_degree);
    }
}

void init_llm(FFModel &tree_model, ModelMeta &model_metadata, 
                GenerationConfig &generationConfig, bool use_full_precision) {
    if (model_metadata.llm_model_type == ModelType::LLAMA) {
        LLAMA::create_llama_model(tree_model,
                                model_metadata.llm_model_config_path,
                                model_metadata.llm_weights_path,
                                TREE_VERIFY_MODE,
                                generationConfig,
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
}

void init_ssms(RequestManager *rm, std::vector<FFModel> &ssm_models, int num_ssms,
                ModelMeta &model_metadata, GenerationConfig &generationConfig,
                bool use_full_precision) {
    for (int ssm_id = 0; ssm_id < num_ssms; ssm_id++) {
        FFModel &beam_model = ssm_models[ssm_id];
        if (model_metadata.ssm_model_types[ssm_id] == ModelType::LLAMA) {
        LLAMA::create_llama_model(beam_model,
                                    model_metadata.ssm_model_config_paths[ssm_id],
                                    model_metadata.ssm_model_weights_paths[ssm_id],
                                    BEAM_SEARCH_MODE,
                                    generationConfig,
                                    use_full_precision);
        } else if (model_metadata.ssm_model_types[ssm_id] == ModelType::OPT) {
        OPT::create_opt_model(beam_model,
                                model_metadata.ssm_model_config_paths[ssm_id],
                                model_metadata.ssm_model_weights_paths[ssm_id],
                                BEAM_SEARCH_MODE,
                                use_full_precision);
        } else if (model_metadata.ssm_model_types[ssm_id] == ModelType::FALCON) {
        FALCON::create_falcon_model(
            beam_model,
            model_metadata.ssm_model_config_paths[ssm_id],
            model_metadata.ssm_model_weights_paths[ssm_id],
            BEAM_SEARCH_MODE,
            use_full_precision);
        } else if (model_metadata.ssm_model_types[ssm_id] == ModelType::MPT) {
        MPT::create_mpt_model(beam_model,
                                model_metadata.ssm_model_config_paths[ssm_id],
                                model_metadata.ssm_model_weights_paths[ssm_id],
                                BEAM_SEARCH_MODE,
                                generationConfig,
                                use_full_precision);
        } else {
        assert(false && "Invalid SSM model type passed.");
        }

        rm->register_ssm_model(&beam_model);
    }
}

json load_trace(std::string input_filename) {
    std::cout << "Loading input file: " << input_filename << std::endl;
    std::ifstream file(input_filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << input_filename << std::endl;
        return nullptr;
    }

    try {
        json data = json::parse(file);
        
        // Print metadata
        const auto& metadata = data["metadata"];
        std::cout << "Metadata:" << std::endl;
        std::cout << "Average entries per partition: " << metadata["avg_entries_per_partition"] << std::endl;
        std::cout << "Max prompt length: " << metadata["max_prompt_length"] << std::endl;
        std::cout << "Min prompt length: " << metadata["min_prompt_length"] << std::endl;
        std::cout << "Avg prompt length: " << metadata["avg_prompt_length"] << std::endl;
        std::cout << "Max response length: " << metadata["max_response_length"] << std::endl;
        std::cout << "Min response length: " << metadata["min_response_length"] << std::endl;
        std::cout << "Avg response length: " << metadata["avg_response_length"] << std::endl;
        // Print list of partition names
        const auto& partitions = data["partitions"];
        std::cout << "Partitions:" << std::endl;
        int counter = 0;
        for (const auto& partition : partitions) {
            std::cout << counter++ << ". " << partition["name"] << std::endl;
        }
    }
    catch (json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return nullptr;
    }
}

json get_training_entries(json data, std::string partition_name) {
    const auto& partitions = data["partitions"];
    for (const auto& partition : partitions) {
        if (partition["name"] == partition_name) {
            return partition["training_entries"];
        }
    }
    std::cerr << "Partition not found: " << partition_name << std::endl;
    return 1;
}

json get_eval_entries(json data, std::string partition_name) {
    const auto& partitions = data["partitions"];
    for (const auto& partition : partitions) {
        if (partition["name"] == partition_name) {
            return partition["eval_entries"];
        }
    }
    std::cerr << "Partition not found: " << partition_name << std::endl;
    return 1;
}