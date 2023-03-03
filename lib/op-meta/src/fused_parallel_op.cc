#include "op-meta/ops/fused_parallel_op.h"
#include "utils/hash-utils.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {

bool operator==(FusedParallelOpAttrs const &lhs, FusedParallelOpAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(FusedParallelOpAttrs const &lhs, FusedParallelOpAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::FusedParallelOpAttrs;

size_t hash<FusedParallelOpAttrs>::operator()(
    FusedParallelOpAttrs const &p) const {
  return visit_hash(p);
}
}
