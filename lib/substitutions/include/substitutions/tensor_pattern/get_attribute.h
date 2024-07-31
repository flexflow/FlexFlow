#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_GET_ATTRIBUTE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_GET_ATTRIBUTE_H

#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_key.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"

namespace FlexFlow {

TensorAttributeValue get_attribute(ParallelTensorAttrs const &,
                                   TensorAttributeKey);

} // namespace FlexFlow

#endif
