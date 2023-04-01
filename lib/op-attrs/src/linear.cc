#include "op-attrs/ops/linear.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(LinearAttrs const &lhs, LinearAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator!=(LinearAttrs const &lhs, LinearAttrs const &rhs) {
  return visit_neq(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::LinearAttrs;

size_t hash<LinearAttrs>::operator()(LinearAttrs const &attrs) const {
  return visit_hash(attrs);
};

}
