#include "op-attrs/ops/softmax.h"
#include "utils/visit_struct.h"

namespace FlexFlow {

bool operator==(SoftmaxAttrs const &lhs, SoftmaxAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(SoftmaxAttrs const &lhs, SoftmaxAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

OperatorType SoftmaxAttrs::op_type() const {
  return OP_SOFTMAX;
}

}

namespace std {
using ::FlexFlow::SoftmaxAttrs;

size_t hash<SoftmaxAttrs>::operator()(SoftmaxAttrs const &params) const {
  return visit_hash(params);
}
}
