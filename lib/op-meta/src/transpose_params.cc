#include "op-meta/ops/transpose_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(TransposeParams const &lhs, TransposeParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(TransposeParams const &lhs, TransposeParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}

namespace std {
using ::FlexFlow::opmeta::TransposeParams;

size_t hash<TransposeParams>::operator()(TransposeParams const &params) const {
  return visit_hash(params);
}
}
