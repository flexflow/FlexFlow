#include "substitutions/tensor_pattern/satisfies_pattern.h"
#include "substitutions/tensor_pattern/satisfies_constraint.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

bool parallel_tensor_satisfies_pattern(ParallelTensorAttrs const &attrs,
                                       TensorAttributePattern const &pattern) {
  return all_of(pattern.attribute_constraints,
                [&](TensorAttributeConstraint const &c) {
                  return parallel_tensor_satisfies_constraint(attrs, c);
                });
}
} // namespace FlexFlow
