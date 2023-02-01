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

bool ElementBinaryParams::is_valid(ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
  size_t numdim = std::min(lhs.num_dims(), rhs.num_dims());
  for (int i = 0; i < numdim; i++) {
    if (lhs.at(i).size > 1 && rhs.at(i).size > 1) {
      if (lhs.at(i) != rhs.at(i)) {
        return false;
      }
    }
  }
  return true;
}

}
