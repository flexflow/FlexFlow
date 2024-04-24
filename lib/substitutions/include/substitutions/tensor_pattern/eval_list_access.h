#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_EVAL_LIST_ACCESS_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_EVAL_LIST_ACCESS_H

#include "substitutions/tensor_pattern/tensor_attribute_list_access.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"
#include "pcg/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {

TensorAttributeValue eval_list_access(ParallelTensorAttrs const &attrs, TensorAttributeListIndexAccess const &);

} // namespace FlexFlow

#endif
