#ifndef _FLEXFLOW_LORA_LINEAR_PARAMS_H
#define _FLEXFLOW_LORA_LINEAR_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_tensor.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace FlexFlow {

class LoraOptimizerConfig {
public:
  LoraOptimizerConfig();
  virtual std::string getType() const = 0;
  virtual nlohmann::json toJson() const = 0;
  static std::unique_ptr<LoraOptimizerConfig> fromJson(const nlohmann::json& j);
  virtual ~LoraOptimizerConfig() = default;
};

class LoraSGDOptimizerConfig : public LoraOptimizerConfig {
public:
  LoraSGDOptimizerConfig();
  LoraSGDOptimizerConfig(double lr_,
                         double momentum_ = 0.0f,
                         bool nesterov_ = false,
                         bool weight_decay_ = 0.0f);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraSGDOptimizerConfig const &llc);
  
  std::string getType() const override { return "SGD"; }  
  
  nlohmann::json toJson() const override {
    return {{"type", "SGD"},
            {"lr", lr},
            {"momentum", momentum},
            {"nesterov", nesterov},
            {"weight_decay", weight_decay}};
  }

  static std::unique_ptr<LoraSGDOptimizerConfig> fromJson(const nlohmann::json& j) {
    auto sgd = std::make_unique<LoraSGDOptimizerConfig>();
    sgd->lr = j["lr"];
    sgd->momentum = j["momentum"];
    sgd->nesterov = j["nesterov"];
    sgd->weight_decay = j["weight_decay"];
    return sgd;
  }

public:
  double lr = 0.001f;
  double momentum = 0.0f;
  bool nesterov = false;
  double weight_decay = 0.0f;
};

class LoraAdamOptimizerConfig : public LoraOptimizerConfig {
public:
  LoraAdamOptimizerConfig();
  LoraAdamOptimizerConfig(double alpha_,
                          double beta1_ = 0.9f,
                          double beta2_ = 0.999f,
                          double weight_decay_ = 0.0f,
                          double epsilon_ = 1e-8);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraAdamOptimizerConfig const &llc);
  
  std::string getType() const override { return "Adam"; }  
  
  nlohmann::json toJson() const override {
    return {{"type", "Adam"},
            {"alpha", alpha},
            {"beta1", beta1},
            {"beta2", beta2},
            {"weight_decay", weight_decay},
            {"epsilon", epsilon}};
  }

  static std::unique_ptr<LoraAdamOptimizerConfig> fromJson(const nlohmann::json& j) {
    auto adam = std::make_unique<LoraAdamOptimizerConfig>();
    adam->alpha = j["alpha"];
    adam->beta1 = j["beta1"];
    adam->beta2 = j["beta2"];
    adam->weight_decay = j["weight_decay"];
    adam->epsilon = j["epsilon"];
    return adam;
  }

public:
  // Adam
  double alpha = 0.001f;
  double beta1 = 0.9f;
  double beta2 = 0.999f;
  double weight_decay = 0.0f;
  double epsilon = 1e-8;
};

std::unique_ptr<LoraOptimizerConfig> LoraOptimizerConfig::fromJson(const nlohmann::json& j) {
  std::string type = j["type"];
  if (type == "SGD") return LoraSGDOptimizerConfig::fromJson(j);
  if (type == "Adam") return LoraAdamOptimizerConfig::fromJson(j);
  throw std::runtime_error("Unknown optimizer type");
}


class LoraLinearConfig {
public:
  static const LoraLinearConfig EmptyConfig;
  LoraLinearConfig(std::string const &cache_folder_,
                   std::string const &peft_model_id_,
                   bool trainable_ = false,
                   LoraOptimizerConfig *optimizer_config_ = nullptr,
                   bool init_lora_weights_ = false,
                   std::string const &base_model_name_or_path_ = "",
                   std::string const &precision_ = "fp16",
                   int rank_ = 8,
                   float lora_alpha_ = 8.0f,
                   float lora_dropout_ = 0.0f,
                   std::vector<std::string> const &target_modules_ = {});
  // constructor used to support std::unordered_map
  LoraLinearConfig();
  template<typename T>
    void setOptimizer(T&& opt) {
        optimizer_config = std::make_unique<T>(std::forward<T>(opt));
    }
  friend bool operator==(LoraLinearConfig const &lhs,
                         LoraLinearConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraLinearConfig const &llc);
  std::string serialize_to_json_string(int indent=-1) const {
    json j = {
        {"cache_folder", cache_folder},
        {"peft_model_id", peft_model_id},
        {"rank", rank},
        {"lora_alpha", lora_alpha},
        {"lora_dropout", lora_dropout},
        {"target_modules", target_modules},
        {"trainable", trainable},
        {"init_lora_weights", init_lora_weights},
        {"base_model_name_or_path", base_model_name_or_path},
        {"precision", precision},
        {"optimizer_config", optimizer_config ? optimizer_config->toJson() : nullptr}
    };

    return j.dump(indent);  // No indentation
  }
  void serialize_to_json_file(const std::string& filename) const {
    std::string j = serialize_to_json_string(4);
    std::ofstream file(filename);
    file << j;
  }
  // Deserialization method
  static LoraLinearConfig deserialize_from_json_string(const std::string& json_string) {
    json j = json::parse(json_string);
    LoraLinearConfig config(
        j["cache_folder"].get<std::string>(),
        j["peft_model_id"].get<std::string>(),
        j["trainable"].get<bool>(),
        nullptr,  // optimizer_config will be set later if present
        j["init_lora_weights"].get<bool>(),
        j["base_model_name_or_path"].get<std::string>(),
        j["precision"].get<std::string>(),
        j["rank"].get<int>(),
        j["lora_alpha"].get<float>(),
        j["lora_dropout"].get<float>(),
        j["target_modules"].get<std::vector<std::string>>()
    );
    if (!j["optimizer_config"].is_null()) {
      config.setOptimizer(LoraOptimizerConfig::fromJson(j["optimizer_config"]));
    }
    return config;
  }
  // Deserialization method
  static LoraLinearConfig deserialize_from_json_file(const std::string& filename) {
    std::ifstream file(filename);
    std::string j;
    file >> j;
    return deserialize_from_json_string(j);
  }

  std::string cache_folder;
  // Huggingface model ID (for download and/or upload)
  std::string peft_model_id;
  // Lora parameters
  int rank;
  float lora_alpha;
  float lora_dropout;
  std::vector<std::string> target_modules;
  // Training parameters
  // whether the weights are trainable (fine-tuning scenario) or not
  // (inference-only). If set to true, allocate space for the gradients
  bool trainable = false;
  // LoraOptimizerConfig *optimizer_config;
  std::unique_ptr<LoraOptimizerConfig> optimizer_config;
  // whether to initialize weights randomly (instead of attempting to load them
  // from file)
  bool init_lora_weights;
  // parameters only used to upload model after finetuning
  std::string base_model_name_or_path;
  std::string precision;
};

class LoraLinearParams {
public:
  LayerID layer_guid;
  OperatorType type;
  // std::unordered_map<PEFTModelID, LoraLinearConfig> peft_configs;
  int max_rank;
  int max_concurrent_adapters;
  char name[MAX_OPNAME];

  bool is_valid(std::pair<ParallelTensorShape, ParallelTensorShape> const
                    &input_shape) const;
  friend bool operator==(LoraLinearParams const &lhs,
                         LoraLinearParams const &rhs);
};

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::LoraLinearParams> {
  size_t operator()(FlexFlow::LoraLinearParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_LORA_LINEAR_PARAMS_H
