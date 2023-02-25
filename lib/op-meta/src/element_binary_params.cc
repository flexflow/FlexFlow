#include "op-meta/ops/element_binary_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(ElementBinaryParams const &lhs, ElementBinaryParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ElementBinaryParams const &lhs, ElementBinaryParams const &rhs) {
  return visit_lt(lhs, rhs);
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
}
