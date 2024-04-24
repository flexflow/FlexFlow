#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "core.h"
#include "op-attrs/ops/element_scalar_unary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementUnaryAttrs const &, ParallelTensorShape const &);
TensorShape get_output_shape(ElementUnaryAttrs const &, TensorShape const &);

ParallelTensorShape get_output_shape(ElementScalarUnaryAttrs const &, ParallelTensorShape const &);
TensorShape get_output_shape(ElementScalarUnaryAttrs const &, TensorShape const &);

CHECK_VALID_OP_ATTR(ElementUnaryAttrs);
CHECK_VALID_OP_ATTR(ElementScalarUnaryAttrs);

} // namespace FlexFlow

#endif
