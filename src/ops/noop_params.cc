#include "flexflow/ops/params/noop_params.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

bool NoOpParams::is_valid(
    std::vector<ParallelTensorShape> const &inputs) const {
  if (this->op_type == OP_NOOP) {
    return inputs.size() == 1;
    /* return inputs[0].is_valid(); */
  } else {
    assert(this->op_type == OP_INPUT);
    return true;
  }
}

bool operator==(NoOpParams const &lhs, NoOpParams const &rhs) {
  return lhs.op_type == rhs.op_type && lhs.input_metadata == rhs.input_metadata;
}

} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::NoOpParams>::operator()(
    FlexFlow::NoOpParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.op_type);
  hash_combine(key, params.input_metadata);
  return key;
}
}; // namespace std
