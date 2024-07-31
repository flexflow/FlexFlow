#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_EVAL_LIST_ACCESS_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_EVAL_LIST_ACCESS_H

#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_list_access.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"

namespace FlexFlow {

TensorAttributeValue eval_list_access(ParallelTensorAttrs const &attrs,
                                      TensorAttributeListIndexAccess const &);

} // namespace FlexFlow

#endif
