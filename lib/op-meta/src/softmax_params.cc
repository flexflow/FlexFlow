#include "op-meta/ops/softmax_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(SoftmaxParams const &lhs, SoftmaxParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(SoftmaxParams const &lhs, SoftmaxParams const &rhs) {
  return visit_lt(lhs, rhs);
}

OperatorType SoftmaxParams::op_type() const {
  return OP_SOFTMAX;
}

}
}

namespace std {
using ::FlexFlow::opmeta::SoftmaxParams;

size_t hash<SoftmaxParams>::operator()(SoftmaxParams const &params) const {
  return visit_hash(params);
}
}
