#include "op-meta/ops/binary_op.h"

namespace FlexFlow {

bool BinaryOpAttrs::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  return inputs.size() == 2 
    && inputs.at(0).is_valid() 
    && inputs.at(1).is_valid()
    && this->is_valid(inputs.at(0), inputs.at(1));
}

bool BinaryOpAttrs::is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const {
  return true;
}

int BinaryOpAttrs::num_outputs(std::vector<ParallelTensorShape> const &inputs) const {
  return 2;
}

}
