#include "pcg/operator.h"

namespace FlexFlow {

Operator::Operator(PCGOperatorAttrs const &attrs,
                   optional<std::string> const &name)
    : attrs(attrs), name(name) {}

Operator::operator PCGOperatorAttrs() const {
  return attrs;
}

} // namespace FlexFlow
