#include "op-meta/ops/dropout.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {

bool operator==(DropoutAttrs const &lhs, DropoutAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(DropoutAttrs const &lhs, DropoutAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {
using ::FlexFlow::DropoutAttrs;

size_t hash<DropoutAttrs>::operator()(
    DropoutAttrs const &params) const {
  return visit_hash(params);
} 
}
