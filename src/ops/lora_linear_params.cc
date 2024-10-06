#include "flexflow/ops/lora_linear_params.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {

// ---------------- Optimizer configs ----------------
// ---------------------------------------------------

// empty optimizer
LoraOptimizerConfig::LoraOptimizerConfig() {}

std::unique_ptr<LoraOptimizerConfig>
    LoraOptimizerConfig::fromJson(nlohmann::json const &j) {
  std::string type = j["type"];
  if (type == "SGD") {
    return LoraSGDOptimizerConfig::fromJson(j);
  }
  if (type == "Adam") {
    return LoraAdamOptimizerConfig::fromJson(j);
  }
  throw std::runtime_error("Unknown optimizer type");
}

// SGD optimizer
LoraSGDOptimizerConfig::LoraSGDOptimizerConfig()
    : lr(0.001f), momentum(0.0f), nesterov(false), weight_decay(0.0f) {}

LoraSGDOptimizerConfig::LoraSGDOptimizerConfig(double lr_,
                                               double momentum_,
                                               bool nesterov_,
                                               bool weight_decay_)
    : lr(lr_), momentum(momentum_), nesterov(nesterov_),
      weight_decay(weight_decay_) {}

std::ostream &operator<<(std::ostream &os, LoraSGDOptimizerConfig const &llc) {
  os << "SGD Optimizer (lr=" << llc.lr << ",momentum=" << llc.momentum
     << ",nesterov=" << llc.nesterov << ",weight_decay=" << llc.weight_decay
     << ")";
  return os;
}

nlohmann::json LoraSGDOptimizerConfig::toJson() const {
  return {{"type", "SGD"},
          {"lr", lr},
          {"momentum", momentum},
          {"nesterov", nesterov},
          {"weight_decay", weight_decay}};
}

std::unique_ptr<LoraSGDOptimizerConfig>
    LoraSGDOptimizerConfig::fromJson(nlohmann::json const &j) {
  auto sgd = std::make_unique<LoraSGDOptimizerConfig>();
  sgd->lr = j["lr"];
  sgd->momentum = j["momentum"];
  sgd->nesterov = j["nesterov"];
  sgd->weight_decay = j["weight_decay"];
  return sgd;
}

// Adam optimizer
LoraAdamOptimizerConfig::LoraAdamOptimizerConfig()
    : alpha(0.001f), beta1(0.9f), beta2(0.999f), weight_decay(0.0f),
      epsilon(1e-8) {}

LoraAdamOptimizerConfig::LoraAdamOptimizerConfig(double alpha_,
                                                 double beta1_,
                                                 double beta2_,
                                                 double weight_decay_,
                                                 double epsilon_)
    : alpha(alpha_), beta1(beta1_), beta2(beta2_), weight_decay(weight_decay_),
      epsilon(epsilon_) {}

std::ostream &operator<<(std::ostream &os, LoraAdamOptimizerConfig const &llc) {
  os << "SGD Optimizer (alpha=" << llc.alpha << ",beta1=" << llc.beta1
     << ",beta2=" << llc.beta2 << ",weight_decay=" << llc.weight_decay
     << ",epsilon=" << llc.epsilon << ")";
  return os;
}

nlohmann::json LoraAdamOptimizerConfig::toJson() const {
  return {{"type", "Adam"},
          {"alpha", alpha},
          {"beta1", beta1},
          {"beta2", beta2},
          {"weight_decay", weight_decay},
          {"epsilon", epsilon}};
}

std::unique_ptr<LoraAdamOptimizerConfig>
    LoraAdamOptimizerConfig::fromJson(nlohmann::json const &j) {
  auto adam = std::make_unique<LoraAdamOptimizerConfig>();
  adam->alpha = j["alpha"];
  adam->beta1 = j["beta1"];
  adam->beta2 = j["beta2"];
  adam->weight_decay = j["weight_decay"];
  adam->epsilon = j["epsilon"];
  return adam;
}

// ------------------ LoRA configs -------------------
// ---------------------------------------------------
const LoraLinearConfig LoraLinearConfig::EmptyConfig = LoraLinearConfig("", "");

LoraLinearConfig::LoraLinearConfig(
    std::string const &cache_folder_,
    std::string const &peft_model_id_,
    bool trainable_,
    LoraOptimizerConfig *optimizer_config_,
    bool init_lora_weights_,
    std::string const &base_model_name_or_path_,
    std::string const &precision_,
    int rank_,
    float lora_alpha_,
    float lora_dropout_,
    std::vector<std::string> const &target_modules_)
    : cache_folder(cache_folder_), peft_model_id(peft_model_id_), rank(rank_),
      lora_alpha(lora_alpha_), lora_dropout(lora_dropout_),
      trainable(trainable_), optimizer_config(optimizer_config_),
      init_lora_weights(init_lora_weights_),
      base_model_name_or_path(base_model_name_or_path_), precision(precision_),
      target_modules(target_modules_) {

  if (peft_model_id.empty()) {
    return;
  }
  assert(!cache_folder.empty() &&
         "cache_folder must be provided when using PEFT");
  if (trainable) {
    assert(optimizer_config != nullptr &&
           "optimizer_config must be provided when using PEFT");
    assert(
        !base_model_name_or_path.empty() &&
        "base_model_name_or_path must be provided when training a PEFT model");
    assert(!precision.empty() &&
           "precision must be provided when training a PEFT model");
  } else {
    assert(init_lora_weights == false &&
           "init_lora_weights must be false when LORA not trainable");
    assert(optimizer_config == nullptr &&
           "optimizer_config must be nullptr when not trainable");
  }
  // if we are not initializing LORA from scratch, load the configs from
  // existing repository
  if (!init_lora_weights) {
    std::string peft_inference_config_file_path =
        join_path({cache_folder, "configs", peft_model_id, "config.json"});
    std::ifstream config_file(peft_inference_config_file_path);
    if (config_file.is_open()) {
      try {
        json model_config;
        config_file >> model_config;
        rank = model_config["r"];
        lora_alpha = float(model_config["lora_alpha"]);
        lora_dropout = model_config["lora_dropout"];
        for (auto &s : model_config["target_modules"]) {
          target_modules.push_back(s);
        }
        // do not load the base_model_name_or_path from the HF config because we
        // may be applying LoRA to another model
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
  assert(rank > 0 && "rank must be greater than 0");
  assert(lora_alpha > 0.0f && "lora_alpha must be greater than 0.0");
  assert(lora_dropout >= 0.0f && lora_dropout <= 1.0f &&
         "lora_dropout must be in [0.0, 1.0]");
  assert(target_modules.size() > 0 && "target_modules must not be left empty");
}

// constructor used to support unordered_map
LoraLinearConfig::LoraLinearConfig() : LoraLinearConfig("", "") {}

bool operator==(LoraLinearConfig const &lhs, LoraLinearConfig const &rhs) {
  if (lhs.cache_folder == rhs.cache_folder &&
      lhs.peft_model_id == rhs.peft_model_id && lhs.rank == rhs.rank &&
      lhs.lora_alpha == rhs.lora_alpha &&
      lhs.lora_dropout == rhs.lora_dropout &&
      lhs.target_modules.size() == rhs.target_modules.size() &&
      lhs.trainable == rhs.trainable &&
      lhs.init_lora_weights == rhs.init_lora_weights &&
      lhs.optimizer_config == rhs.optimizer_config &&
      lhs.base_model_name_or_path == rhs.base_model_name_or_path &&
      lhs.precision == rhs.precision) {
    for (int i = 0; i < lhs.target_modules.size(); i++) {
      if (lhs.target_modules[i] != rhs.target_modules[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, LoraLinearConfig const &llc) {
  os << "LoraLinearConfig: ";
  os << "cache_folder: " << llc.cache_folder << ", ";
  os << "peft_model_id: " << llc.peft_model_id << ", ";
  os << "rank: " << llc.rank << ", ";
  os << "lora_alpha: " << llc.lora_alpha << ", ";
  os << "lora_dropout: " << llc.lora_dropout << ", ";
  os << "target_modules: [";
  for (int i = 0; i < llc.target_modules.size(); i++) {
    os << llc.target_modules[i];
    if (i < llc.target_modules.size() - 1) {
      os << ", ";
    }
  }
  os << "], ";
  os << "trainable: " << llc.trainable << ", ";
  if (llc.optimizer_config != nullptr) {
    os << "optimizer_config: ";
    if (llc.optimizer_config.get()->getType() == "SGD") {
      os << *static_cast<LoraSGDOptimizerConfig const *>(
          llc.optimizer_config.get());
    } else if (llc.optimizer_config.get()->getType() == "Adam") {
      os << *static_cast<LoraAdamOptimizerConfig const *>(
          llc.optimizer_config.get());
    } else {
      os << "Unknown optimizer config type";
    }
    std::cout << std::endl;
  }
  os << "init_lora_weights: " << llc.init_lora_weights << std::endl;
  os << "base_model_name_or_path: " << llc.base_model_name_or_path << std::endl;
  os << "precision: " << llc.precision << std::endl;
  return os;
}

std::string LoraLinearConfig::serialize_to_json_string(int indent) const {
  nlohmann::json j = {{"cache_folder", cache_folder},
                      {"peft_model_id", peft_model_id},
                      {"rank", rank},
                      {"lora_alpha", lora_alpha},
                      {"lora_dropout", lora_dropout},
                      {"target_modules", target_modules},
                      {"trainable", trainable},
                      {"init_lora_weights", init_lora_weights},
                      {"base_model_name_or_path", base_model_name_or_path},
                      {"precision", precision},
                      // {"optimizer_config", optimizer_config ?
                      // optimizer_config->toJson() : nullptr}
                      {"optimizer_config",
                       optimizer_config
                           ? nlohmann::json(optimizer_config->toJson())
                           : nlohmann::json()}};

  return j.dump(indent); // No indentation
}

void LoraLinearConfig::serialize_to_json_file(
    std::string const &filename) const {
  std::string j = serialize_to_json_string(4);
  std::ofstream file(filename);
  file << j;
}

// Deserialization method
LoraLinearConfig LoraLinearConfig::deserialize_from_json_string(
    std::string const &json_string) {
  nlohmann::json j = nlohmann::json::parse(json_string);
  LoraLinearConfig config(
      j["cache_folder"].get<std::string>(),
      j["peft_model_id"].get<std::string>(),
      j["trainable"].get<bool>(),
      nullptr, // optimizer_config will be set later if present
      j["init_lora_weights"].get<bool>(),
      j["base_model_name_or_path"].get<std::string>(),
      j["precision"].get<std::string>(),
      j["rank"].get<int>(),
      j["lora_alpha"].get<float>(),
      j["lora_dropout"].get<float>(),
      j["target_modules"].get<std::vector<std::string>>());
  if (!j["optimizer_config"].is_null()) {
    config.setOptimizer(LoraOptimizerConfig::fromJson(j["optimizer_config"]));
  }
  return config;
}

// Deserialization method
LoraLinearConfig
    LoraLinearConfig::deserialize_from_json_file(std::string const &filename) {
  std::ifstream file(filename);
  std::string j;
  file >> j;
  return deserialize_from_json_string(j);
}

}; // namespace FlexFlow
