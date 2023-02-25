#include "op-meta/ops/replicate_params.h"
#include "utils/hash-utils.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(ReplicateParams const &lhs, ReplicateParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ReplicateParams const &lhs, ReplicateParams const &rhs) {
  return visit_lt(lhs, rhs);
}

OperatorType ReplicateParams::op_type() const {
  return OP_REPLICATE;
}

}
}

namespace std {
using ::FlexFlow::opmeta::ReplicateParams;

size_t hash<ReplicateParams>::operator()(
    ReplicateParams const &params) const {
  return visit_hash(params);
} 
}
