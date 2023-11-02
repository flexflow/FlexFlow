#include "flexflow/ops/lora_linear_params.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {
const LoraLinearConfig LoraLinearConfig::DefaultConfig = LoraLinearConfig();

LoraLinearConfig::LoraLinearConfig()
    : rank(0), optimizer_type(OPTIMIZER_TYPE_NONE), learning_rate(0.0f),
      cache_folder(""), peft_model_id(""), lora_alpha(0), lora_dropout(0.0f),
      load_weights_from_file(false) {}

LoraLinearConfig::LoraLinearConfig(int _rank, OptimizerType _type, float _lr)
    : rank(_rank), optimizer_type(_type), learning_rate(_lr), cache_folder(""),
      peft_model_id(""), lora_alpha(0), lora_dropout(0.0f),
      load_weights_from_file(false) {}

LoraLinearConfig::LoraLinearConfig(std::string const &cache_folder_,
                                   std::string const &peft_model_id_) {
  cache_folder = cache_folder_;
  peft_model_id = peft_model_id_;
  std::string peft_inference_config_file_path =
      join_path({cache_folder, "configs", peft_model_id, "config.json"});
  std::ifstream config_file(peft_inference_config_file_path);
  if (config_file.is_open()) {
    try {
      json model_config;
      config_file >> model_config;
      rank = model_config["r"];
      lora_alpha = model_config["lora_alpha"];
      lora_dropout = model_config["lora_dropout"];
    } catch (json::exception const &e) {
      std::cerr << "Error parsing PEFT config from JSON file: " << e.what()
                << std::endl;
      assert(false);
    }
  } else {
    std::cerr << "Error opening JSON file " << peft_inference_config_file_path
              << std::endl;
    assert(false);
  }
  optimizer_type = OPTIMIZER_TYPE_NONE;
  learning_rate = 0.0f;
  load_weights_from_file = true;
}

bool operator==(LoraLinearConfig const &lhs, LoraLinearConfig const &rhs) {
  if (lhs.rank == rhs.rank && lhs.optimizer_type == rhs.optimizer_type &&
      lhs.learning_rate == rhs.learning_rate) {
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraLinearConfig const &llc) {
  os << "LoraLinearConfig: ";
  os << "rank: " << llc.rank << ", ";
  os << "optimizer_type: " << llc.optimizer_type << ", ";
  os << "learning_rate: " << llc.learning_rate << ", ";
  os << "cache_folder: " << llc.cache_folder << ", ";
  os << "peft_model_id: " << llc.peft_model_id << ", ";
  os << "lora_alpha: " << llc.lora_alpha << ", ";
  os << "lora_dropout: " << llc.lora_dropout << ", ";
  os << "load_weights_from_file: " << llc.load_weights_from_file << std::endl;
  return os;
}

}; // namespace FlexFlow
