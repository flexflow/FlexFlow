#include "op-attrs/ops/element_unary.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(ElementUnaryAttrs const &lhs, ElementUnaryAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ElementUnaryAttrs const &lhs, ElementUnaryAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {

using ::FlexFlow::ElementUnaryAttrs;

size_t hash<ElementUnaryAttrs>::operator()(ElementUnaryAttrs const &attrs) const {
  return visit_hash(attrs);
}

}
