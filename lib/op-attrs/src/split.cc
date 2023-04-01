#include "op-attrs/ops/split.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

OperatorType SplitAttrs::op_type() const {
  return OP_SPLIT;
}

bool operator==(SplitAttrs const &lhs, SplitAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(SplitAttrs const &lhs, SplitAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::SplitAttrs;

size_t hash<SplitAttrs>::operator()(SplitAttrs const &p) const {
  return visit_hash(p);
}
}
