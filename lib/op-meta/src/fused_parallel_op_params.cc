#include "op-meta/ops/fused_parallel_op_params.h"
#include "utils/hash-utils.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(FusedParallelOpParams const &lhs, FusedParallelOpParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(FusedParallelOpParams const &lhs, FusedParallelOpParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::FusedParallelOpParams;

size_t hash<FusedParallelOpParams>::operator()(
    FusedParallelOpParams const &p) const {
  return visit_hash(p);
}
}
