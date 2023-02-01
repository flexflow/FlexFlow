#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

bool UnaryOpParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return (inputs.size() == 1 && inputs.at(0).is_valid() && this->is_valid(inputs.at(0)));
}

bool UnaryOpParams::is_valid(ParallelTensorShape const &) const {
  return true;
}

int UnaryOpParams::num_outputs(std::vector<ParallelTensorShape> const &inputs) const {
  return 1;
}

}
