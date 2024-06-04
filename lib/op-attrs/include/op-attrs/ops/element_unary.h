#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/element_scalar_unary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(ElementUnaryAttrs const &, TensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ElementUnaryAttrs const &, ParallelTensorShape const &);

tl::expected<TensorShape, std::string>
    get_output_shape(ElementScalarUnaryAttrs const &, TensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ElementScalarUnaryAttrs const &,
                     ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(ElementUnaryAttrs);
CHECK_VALID_OP_ATTR(ElementScalarUnaryAttrs);

using ElementUnaryUnifiedAttrs =
    std::variant<ElementUnaryAttrs, ElementScalarUnaryAttrs>;

} // namespace FlexFlow

#endif
