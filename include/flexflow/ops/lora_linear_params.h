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
  static std::unique_ptr<LoraOptimizerConfig> fromJson(nlohmann::json const &j);
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

  std::string getType() const override {
    return "SGD";
  }

  nlohmann::json toJson() const override;

  static std::unique_ptr<LoraSGDOptimizerConfig>
      fromJson(nlohmann::json const &j);

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

  std::string getType() const override {
    return "Adam";
  }

  nlohmann::json toJson() const override;

  static std::unique_ptr<LoraAdamOptimizerConfig>
      fromJson(nlohmann::json const &j);

public:
  // Adam
  double alpha = 0.001f;
  double beta1 = 0.9f;
  double beta2 = 0.999f;
  double weight_decay = 0.0f;
  double epsilon = 1e-8;
};

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

  // Method to set optimizer
  template <typename T>
  void setOptimizer(T &&opt) {
    if constexpr (std::is_base_of_v<LoraOptimizerConfig,
                                    std::remove_reference_t<T>>) {
      optimizer_config =
          std::make_unique<std::remove_reference_t<T>>(std::forward<T>(opt));
    } else if constexpr (std::is_same_v<std::unique_ptr<LoraOptimizerConfig>,
                                        std::remove_reference_t<T>>) {
      optimizer_config = std::move(opt);
    } else {
      static_assert(always_false<T>, "Unsupported optimizer type");
    }
  }
  // Helper template for static_assert
  template <typename>
  static inline constexpr bool always_false = false;

  friend bool operator==(LoraLinearConfig const &lhs,
                         LoraLinearConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraLinearConfig const &llc);
  std::string serialize_to_json_string(int indent = -1) const;
  void serialize_to_json_file(std::string const &filename) const;
  // Deserialization method
  static LoraLinearConfig
      deserialize_from_json_string(std::string const &json_string);
  // Deserialization method
  static LoraLinearConfig
      deserialize_from_json_file(std::string const &filename);

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
