#include "op-meta/ops/dropout_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(DropoutParams const &lhs, DropoutParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(DropoutParams const &lhs, DropoutParams const &rhs) {
  return visit_lt(lhs, rhs);
}
}
}

namespace std {
using ::FlexFlow::opmeta::DropoutParams;

size_t hash<DropoutParams>::operator()(
    DropoutParams const &params) const {
  return visit_hash(params);
} 
}
