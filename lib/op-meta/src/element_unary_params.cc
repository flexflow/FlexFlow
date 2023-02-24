#include "op-meta/ops/element_unary_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

bool operator==(ElementUnaryParams const &lhs, ElementUnaryParams const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator<(ElementUnaryParams const &lhs, ElementUnaryParams const &rhs) {
  return visit_lt(lhs, rhs);
}

}
}
