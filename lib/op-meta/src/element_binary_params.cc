#include "op-meta/ops/element_binary_params.h"

namespace FlexFlow {

typename ElementBinaryParams::AsConstTuple ElementBinaryParams::as_tuple() const {
  return {this->type};
}

bool operator==(ElementBinaryParams const &lhs, ElementBinaryParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(ElementBinaryParams const &lhs, ElementBinaryParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

bool ElementBinaryParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  bool is_valid = true;
  is_valid &= (input.first.is_valid() & input.second.is_valid());
  if (!is_valid) {
    return false;
  }
  // is_valid &= (input.first == input.second);
  ParallelTensorShape lhs = input.first;
  ParallelTensorShape rhs = input.second;
  size_t numdim = std::min(lhs.num_dims(), rhs.num_dims());
  for (int i = 0; i < numdim; i++) {
    if (lhs.at(i).size > 1 && rhs.at(i).size > 1) {
      if (lhs.at(i) != rhs.at(i)) {
        return false;
      }
    }
  }
  return is_valid;
}

}
