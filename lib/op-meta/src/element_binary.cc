#include "op-meta/ops/element_binary.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {

bool operator==(ElementBinaryAttrs const &lhs, ElementBinaryAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ElementBinaryAttrs const &lhs, ElementBinaryAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

bool ElementBinaryAttrs::is_valid(ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
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

namespace std {

using ::FlexFlow::ElementBinaryAttrs;

size_t hash<ElementBinaryAttrs>::operator()(ElementBinaryAttrs const &attrs) const {
  return visit_hash(attrs);
}

}
