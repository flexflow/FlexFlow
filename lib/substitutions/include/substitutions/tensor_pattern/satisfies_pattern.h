#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_SATISFIES_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_SATISFIES_PATTERN_H

#include "substitutions/tensor_pattern/tensor_attribute_pattern.dtg.h"
#include "pcg/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {

bool parallel_tensor_satisfies_pattern(ParallelTensorAttrs const &attrs, TensorAttributePattern const &pattern);

} // namespace FlexFlow

#endif
