#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_GET_ATTRIBUTE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_GET_ATTRIBUTE_H

#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_key.dtg.h"
#include "pcg/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {

TensorAttributeValue get_attribute(ParallelTensorAttrs const &, TensorAttributeKey);

} // namespace FlexFlow

#endif
