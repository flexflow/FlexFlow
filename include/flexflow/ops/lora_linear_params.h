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
  friend bool operator==(LoraOptimizerConfig const &lhs,
                         LoraOptimizerConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraOptimizerConfig const &llc);

public:
  OptimizerType type = OPTIMIZER_TYPE_NONE;
};

class LoraSGDOptimizerConfig : public LoraOptimizerConfig {
public:
  LoraSGDOptimizerConfig();
  LoraSGDOptimizerConfig(double lr_,
                         double momentum_ = 0.0f,
                         bool nesterov_ = false,
                         bool weight_decay_ = 0.0f);
  friend bool operator==(LoraSGDOptimizerConfig const &lhs,
                         LoraSGDOptimizerConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraSGDOptimizerConfig const &llc);

public:
  OptimizerType type = OPTIMIZER_TYPE_SGD;
  double lr = 0.01f;
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
  friend bool operator==(LoraAdamOptimizerConfig const &lhs,
                         LoraAdamOptimizerConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraAdamOptimizerConfig const &llc);

public:
  OptimizerType type = OPTIMIZER_TYPE_ADAM;
  // Adam
  double alpha = 0.001f;
  double beta1 = 0.9f;
  double beta2 = 0.999f;
  double weight_decay = 0.0f;
  double epsilon = 1e-8;
};

class LoraLinearConfig {
public:
  static const LoraLinearConfig DefaultConfig;
  LoraLinearConfig();
  LoraLinearConfig(int _rank,
                   bool _trainable = false,
                   LoraOptimizerConfig *_optimizer_config = nullptr);
  LoraLinearConfig(std::string const &cache_folder_,
                   std::string const &peft_model_id_,
                   bool trainable_ = false,
                   LoraOptimizerConfig *optimizer_config_ = nullptr);
  friend bool operator==(LoraLinearConfig const &lhs,
                         LoraLinearConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraLinearConfig const &llc);

public:
  int rank;
  // whether the weights are trainable (fine-tuning scenario) or not
  // (inference-only). If set to true, allocate space for the gradients
  bool trainable = false;
  LoraOptimizerConfig *optimizer_config;
  std::string cache_folder;
  // Huggingface
  std::string peft_model_id;
  int lora_alpha;
  float lora_dropout;
  // whether to load weights from file, instead of initializing them randomly
  bool load_weights_from_file;
};

class LoraLinearParams {
public:
  LayerID layer_guid;
  OperatorType type;
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
