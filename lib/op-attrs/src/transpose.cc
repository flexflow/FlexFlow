#include "op-attrs/ops/transpose.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(TransposeAttrs const &lhs, TransposeAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(TransposeAttrs const &lhs, TransposeAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::TransposeAttrs;

size_t hash<TransposeAttrs>::operator()(TransposeAttrs const &params) const {
  return visit_hash(params);
}
}
