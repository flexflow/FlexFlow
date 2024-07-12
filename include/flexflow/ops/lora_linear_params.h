#ifndef _FLEXFLOW_LORA_LINEAR_PARAMS_H
#define _FLEXFLOW_LORA_LINEAR_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

class LoraOptimizerConfig {
public:
  LoraOptimizerConfig();
  virtual ~LoraOptimizerConfig() {}
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
                   int rank_ = 8,
                   float lora_alpha_ = 8.0f,
                   float lora_dropout_ = 0.0f,
                   std::vector<std::string> const &target_modules_ = {});
  // constructor used to support std::unordered_map
  LoraLinearConfig();
  friend bool operator==(LoraLinearConfig const &lhs,
                         LoraLinearConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraLinearConfig const &llc);

public:
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
  LoraOptimizerConfig *optimizer_config;
  // whether to initialize weights randomly (instead of attempting to load them
  // from file)
  bool init_lora_weights;
};

class LoraLinearParams {
public:
  LayerID layer_guid;
  OperatorType type;
  std::unordered_map<PEFTModelID, LoraLinearConfig> peft_configs;
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
