#include "flexflow/ops/lora_linear_params.h"

namespace FlexFlow {
const LoraLinearConfig LoraLinearConfig::DefaultConfig = LoraLinearConfig();

LoraLinearConfig::LoraLinearConfig()
    : rank(0), optimizer_type(OPTIMIZER_TYPE_NONE), learning_rate(0.0f) {}

LoraLinearConfig::LoraLinearConfig(int _rank, OptimizerType _type, float _lr)
    : rank(_rank), optimizer_type(_type), learning_rate(_lr) {}

bool operator==(LoraLinearConfig const &lhs, LoraLinearConfig const &rhs) {
  if (lhs.rank == rhs.rank && lhs.optimizer_type == rhs.optimizer_type &&
      lhs.learning_rate == rhs.learning_rate) {
    return true;
  }
  return false;
}

}; // namespace FlexFlow
