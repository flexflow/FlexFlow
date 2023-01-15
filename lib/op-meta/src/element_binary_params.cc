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
  ParallelTensorShape A = input.first;
  ParallelTensorShape B = input.second;
  int numdim = std::min(A.num_dims, B.num_dims);
  for (int i = 0; i < numdim; i++) {
    if (A.dims[i].size > 1 && B.dims[i].size > 1) {
      if (A.dims[i] != B.dims[i]) {
        return false;
      }
    }
  }
  return is_valid;
}

}
