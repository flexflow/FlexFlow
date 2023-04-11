#include "op-attrs/ops/softmax.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(SoftmaxAttrs const &lhs, SoftmaxAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator!=(SoftmaxAttrs const &lhs, SoftmaxAttrs const &rhs) {
  return visit_neq(lhs, rhs);
}

bool operator<(SoftmaxAttrs const &lhs, SoftmaxAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::SoftmaxAttrs;

size_t hash<SoftmaxAttrs>::operator()(SoftmaxAttrs const &params) const {
  return visit_hash(params);
}
}
