#ifndef _FLEXFLOW_LORA_LINEAR_PARAMS_H
#define _FLEXFLOW_LORA_LINEAR_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/inference.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

class LoraLinearConfig {
public:
  static const LoraLinearConfig DefaultConfig;
  LoraLinearConfig();
  LoraLinearConfig(int rank,
                   OptimizerType type = OPTIMIZER_TYPE_SGD,
                   float learning_rate = 1e-4);
  LoraLinearConfig(std::string const &cache_folder_,
                   std::string const &peft_model_id_);
  friend bool operator==(LoraLinearConfig const &lhs,
                         LoraLinearConfig const &rhs);
  friend std::ostream &operator<<(std::ostream &os,
                                  LoraLinearConfig const &llc);

public:
  int rank;
  OptimizerType optimizer_type;
  float learning_rate;
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
