#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/record_formatter.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(LinearAttrs);

RecordFormatter as_dot(LinearAttrs const &);

tl::expected<TensorShape, std::string>
    get_kernel_shape(LinearAttrs const &attrs, TensorShape const &input);
tl::expected<TensorShape, std::string> get_bias_shape(LinearAttrs const &attrs,
                                                      TensorShape const &input);
tl::expected<TensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs, TensorShape const &input);

tl::expected<ParallelTensorShape, std::string>
    get_kernel_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input);
tl::expected<ParallelTensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, ParallelTensorShape const &input);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input);

} // namespace FlexFlow

#endif
