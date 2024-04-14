#include "pcg/operator.h"

namespace FlexFlow {

Operator::operator PCGOperatorAttrs() const {
  return attrs;
}

} // namespace FlexFlow
