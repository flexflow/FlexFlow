#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_EVAL_LIST_SIZE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_EVAL_LIST_SIZE_H

#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_list_size.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"

namespace FlexFlow {

TensorAttributeValue eval_list_size(ParallelTensorAttrs const &attrs,
                                    TensorAttributeListSize const &);

} // namespace FlexFlow

#endif
