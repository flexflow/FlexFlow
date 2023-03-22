#include "op-attrs/ops/concat.h"
#include "utils/visit_struct.h"

namespace FlexFlow {

bool operator==(ConcatAttrs const &lhs, ConcatAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ConcatAttrs const &lhs, ConcatAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

bool ConcatAttrs::is_valid(
    std::vector<ParallelTensorShape> const &input) const {
  bool valid = true;
  for (auto p : input) {
    valid &= p.is_valid();
  }
  return valid;
}

}

namespace std {
using ::FlexFlow::ConcatAttrs;

size_t hash<ConcatAttrs>::operator()(ConcatAttrs const &p) const {
  return visit_hash(p);
}

}
