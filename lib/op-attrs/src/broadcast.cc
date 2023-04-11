#include "op-attrs/ops/broadcast.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

BroadcastAttrs::BroadcastAttrs(stack_vector<int, MAX_TENSOR_DIM> const &target_dims)
  : target_dims(target_dims)
{ }

bool operator==(BroadcastAttrs const &lhs, BroadcastAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator!=(BroadcastAttrs const &lhs, BroadcastAttrs const &rhs) {
  return visit_neq(lhs, rhs);
}

bool operator<(BroadcastAttrs const &lhs, BroadcastAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {

using ::FlexFlow::BroadcastAttrs;

size_t hash<BroadcastAttrs>::operator()(BroadcastAttrs const &attrs) const { return visit_hash(attrs); }

}
