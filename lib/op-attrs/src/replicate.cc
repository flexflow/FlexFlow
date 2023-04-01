#include "op-attrs/ops/replicate.h"
#include "utils/hash-utils.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(ReplicateAttrs const &lhs, ReplicateAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ReplicateAttrs const &lhs, ReplicateAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

/* OperatorType ReplicateAttrs::op_type() const { */
/*   return OP_REPLICATE; */
/* } */

}

namespace std {
using ::FlexFlow::ReplicateAttrs;

size_t hash<ReplicateAttrs>::operator()(
    ReplicateAttrs const &params) const {
  return visit_hash(params);
} 
}
