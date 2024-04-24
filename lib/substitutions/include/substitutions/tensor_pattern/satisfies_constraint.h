#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_SATISFIES_CONSTRAINT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_SATISFIES_CONSTRAINT_H

#include "substitutions/tensor_pattern/tensor_attribute_constraint.dtg.h"
#include "pcg/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {

bool parallel_tensor_satisfies_constraint(ParallelTensorAttrs const &params, TensorAttributeConstraint const &constraint);

} // namespace FlexFlow

#endif
