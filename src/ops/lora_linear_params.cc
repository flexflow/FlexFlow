#include "flexflow/ops/lora_linear_params.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {
const LoraLinearConfig LoraLinearConfig::EmptyConfig = LoraLinearConfig();

LoraLinearConfig::LoraLinearConfig()
    : rank(0), optimizer_type(OPTIMIZER_TYPE_NONE), learning_rate(0.0f),
      config_folder(""), peft_model_id(""), lora_alpha(0), lora_dropout(0.0f),
      load_weights_from_file(false) {}

LoraLinearConfig::LoraLinearConfig(int _rank, OptimizerType _type, float _lr)
    : rank(_rank), optimizer_type(_type), learning_rate(_lr), config_folder(""),
      peft_model_id(""), lora_alpha(0), lora_dropout(0.0f),
      load_weights_from_file(false) {}

LoraLinearConfig::LoraLinearConfig(std::string const &config_folder_,
                                   std::string const &peft_model_id_) {
  config_folder = config_folder_;
  peft_model_id = peft_model_id_;
  std::string peft_inference_config_file_path =
      join_path({config_folder, peft_model_id, "config.json"});
  std::ifstream config_file(peft_inference_config_file_path);
  if (config_file.is_open()) {
    try {
      json model_config;
      config_file >> model_config;
      rank = model_config["r"];
      lora_alpha = model_config["lora_alpha"];
      lora_dropout = model_config["lora_dropout"];
      for (auto &s : model_config["target_modules"]) {
        target_modules.push_back(s); 
      }
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
      lhs.learning_rate == rhs.learning_rate && lhs.config_folder == rhs.config_folder &&
      lhs.peft_model_id == rhs.peft_model_id && lhs.lora_alpha == rhs.lora_alpha &&
      lhs.lora_dropout == rhs.lora_dropout && lhs.target_modules.size() == rhs.target_modules.size() &&
      lhs.load_weights_from_file == rhs.load_weights_from_file) {
    for (int i=0; i<lhs.target_modules.size(); i++) {
      if (lhs.target_modules[i] != rhs.target_modules[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraLinearConfig const &llc) {
  os << "LoraLinearConfig: {";
  os << "rank: " << llc.rank << ", ";
  os << "optimizer_type: " << llc.optimizer_type << ", ";
  os << "learning_rate: " << llc.learning_rate << ", ";
  os << "config_folder: " << llc.config_folder << ", ";
  os << "peft_model_id: " << llc.peft_model_id << ", ";
  os << "lora_alpha: " << llc.lora_alpha << ", ";
  os << "lora_dropout: " << llc.lora_dropout << ", ";
  os << "target_modules: [";
  for (int i=0; i<llc.target_modules.size(); i++) {
    os << llc.target_modules[i];
    if (i < llc.target_modules.size() - 1) {
      os << ", ";
    }
  }
  os << "], ";
  os << "load_weights_from_file: " << llc.load_weights_from_file << std::endl;
  return os;
}

}; // namespace FlexFlow
