#include "local-execution/op_tensor_spec.h"

namespace FlexFlow {

OpTensorSpec input_tensor(int idx, OpSlotOptions option) {
  return {TensorRole::INPUT, option, idx};
}

OpTensorSpec output_tensor(int idx, OpSlotOptions option) {
  return {TensorRole::OUTPUT, option, idx};
}

OpTensorSpec weight_tensor(int idx, OpSlotOptions option) {
  return {TensorRole::WEIGHT, option, idx};
}

} // namespace FlexFlow
