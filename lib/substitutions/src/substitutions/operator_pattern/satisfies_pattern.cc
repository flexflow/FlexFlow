#include "substitutions/operator_pattern/satisfies_pattern.h"
#include "substitutions/operator_pattern/satisfies_constraint.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

bool operator_satisfies_pattern(PCGOperatorAttrs const &attrs,
                                OperatorAttributePattern const &pattern) {
  return all_of(pattern.attribute_constraints,
                [&](OperatorAttributeConstraint const &c) {
                  return operator_satisfies_constraint(attrs, c);
                });
}

} // namespace FlexFlow
