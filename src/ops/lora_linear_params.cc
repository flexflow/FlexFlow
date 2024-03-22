#include "flexflow/ops/lora_linear_params.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {

// ---------------- Optimizer configs ----------------
// ---------------------------------------------------

// empty optimizer
LoraOptimizerConfig::LoraOptimizerConfig() : type(OPTIMIZER_TYPE_NONE) {}

bool operator==(LoraOptimizerConfig const &lhs,
                LoraOptimizerConfig const &rhs) {
  if (lhs.type == rhs.type) {
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraOptimizerConfig const &llc) {
  os << "No Optimizer";
  return os;
}

// SGD optimizer
LoraSGDOptimizerConfig::LoraSGDOptimizerConfig()
    : type(OPTIMIZER_TYPE_SGD), lr(0.01f), momentum(0.0f), nesterov(false),
      weight_decay(0.0f) {}

LoraSGDOptimizerConfig::LoraSGDOptimizerConfig(double lr_,
                                               double momentum_,
                                               bool nesterov_,
                                               bool weight_decay_)
    : type(OPTIMIZER_TYPE_SGD), lr(lr_), momentum(momentum_),
      nesterov(nesterov_), weight_decay(weight_decay_) {}

bool operator==(LoraSGDOptimizerConfig const &lhs,
                LoraSGDOptimizerConfig const &rhs) {
  if (lhs.type == rhs.type && lhs.lr == rhs.lr &&
      lhs.momentum == rhs.momentum && lhs.nesterov == rhs.nesterov &&
      lhs.weight_decay == rhs.weight_decay) {
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraSGDOptimizerConfig const &llc) {
  os << "SGD Optimizer (lr=" << llc.lr << ",momentum=" << llc.momentum
     << ",nesterov=" << llc.nesterov << ",weight_decay=" << llc.weight_decay
     << ")";
  return os;
}

// Adam optimizer
LoraAdamOptimizerConfig::LoraAdamOptimizerConfig()
    : type(OPTIMIZER_TYPE_ADAM), alpha(0.001f), beta1(0.9f), beta2(0.999f),
      weight_decay(0.0f), epsilon(1e-8) {}

LoraAdamOptimizerConfig::LoraAdamOptimizerConfig(double alpha_,
                                                 double beta1_,
                                                 double beta2_,
                                                 double weight_decay_,
                                                 double epsilon_)
    : type(OPTIMIZER_TYPE_ADAM), alpha(alpha_), beta1(beta1_), beta2(beta2_),
      weight_decay(weight_decay_), epsilon(epsilon_) {}

bool operator==(LoraAdamOptimizerConfig const &lhs,
                LoraAdamOptimizerConfig const &rhs) {
  if (lhs.type == rhs.type && lhs.alpha == rhs.alpha &&
      lhs.beta1 == rhs.beta1 && lhs.beta2 == rhs.beta2 &&
      lhs.weight_decay == rhs.weight_decay && lhs.epsilon == rhs.epsilon) {
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraAdamOptimizerConfig const &llc) {
  os << "SGD Optimizer (alpha=" << llc.alpha << ",beta1=" << llc.beta1
     << ",beta2=" << llc.beta2 << ",weight_decay=" << llc.weight_decay
     << ",epsilon=" << llc.epsilon << ")";
  return os;
}

// ------------------ LoRA configs -------------------
// ---------------------------------------------------
const LoraLinearConfig LoraLinearConfig::DefaultConfig = LoraLinearConfig();

LoraLinearConfig::LoraLinearConfig()
    : rank(0), trainable(false), optimizer_config(nullptr), cache_folder(""),
      peft_model_id(""), lora_alpha(0), lora_dropout(0.0f),
      load_weights_from_file(false) {}

LoraLinearConfig::LoraLinearConfig(int _rank,
                                   bool _trainable,
                                   LoraOptimizerConfig *_optimizer_config)
    : rank(_rank), trainable(_trainable), optimizer_config(_optimizer_config),
      cache_folder(""), peft_model_id(""), lora_alpha(0), lora_dropout(0.0f),
      load_weights_from_file(false) {}

LoraLinearConfig::LoraLinearConfig(std::string const &cache_folder_,
                                   std::string const &peft_model_id_,
                                   bool trainable_,
                                   LoraOptimizerConfig *optimizer_config_)
    : cache_folder(cache_folder_), peft_model_id(peft_model_id_),
      trainable(trainable_), optimizer_config(optimizer_config_),
      load_weights_from_file(true) {
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
}

bool operator==(LoraLinearConfig const &lhs, LoraLinearConfig const &rhs) {
  if (lhs.rank == rhs.rank && lhs.optimizer_config == rhs.optimizer_config) {
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraLinearConfig const &llc) {
  os << "LoraLinearConfig: ";
  os << "trainable: " << llc.trainable << ", ";
  os << "rank: " << llc.rank << ", ";
  if (!llc.optimizer_config) {
    os << "optimizer_config: " << llc.optimizer_config << ", ";
  } else {
    os << "optimizer_config: " << *llc.optimizer_config << ", ";
  }
  os << "cache_folder: " << llc.cache_folder << ", ";
  os << "peft_model_id: " << llc.peft_model_id << ", ";
  os << "lora_alpha: " << llc.lora_alpha << ", ";
  os << "lora_dropout: " << llc.lora_dropout << ", ";
  os << "load_weights_from_file: " << llc.load_weights_from_file << std::endl;
  return os;
}

}; // namespace FlexFlow
