#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <tl/expected.hpp> 

namespace FlexFlow {

tl::expected<TensorShape, std::string> 
  get_output_shape(ElementBinaryAttrs const &,
                             TensorShape const &,
                             TensorShape const &);
tl::expected<ParallelTensorShape, std::string> get_output_shape(ElementBinaryAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(ElementBinaryAttrs);

} // namespace FlexFlow

#endif
