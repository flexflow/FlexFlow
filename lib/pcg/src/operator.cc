#include "pcg/operator.h"

namespace FlexFlow {

Operator::Operator(PCGOperatorAttrs const &attrs,
                   std::optional<std::string> const &name)
    : attrs(attrs) {}

Operator::operator PCGOperatorAttrs() const {
  return attrs;
}

} // namespace FlexFlow
