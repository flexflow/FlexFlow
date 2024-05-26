#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "core.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &);
TensorShape get_output_shape(ElementBinaryAttrs const &,
                             TensorShape const &,
                             TensorShape const &);

CHECK_VALID_OP_ATTR(ElementBinaryAttrs);

} // namespace FlexFlow

#endif
