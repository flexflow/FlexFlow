#ifndef _FLEXFLOW_INCLUDE_OP_ATTRS_GET_OUTPUT_SHAPES_H
#define _FLEXFLOW_INCLUDE_OP_ATTRS_GET_OUTPUT_SHAPES_H

#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <vector>

namespace FlexFlow {

std::vector<ParallelTensorShape> get_output_shapes(PCGOperatorAttrs const &,
                                                   std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
