#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_MATMUL_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_MATMUL_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/batch_matmul.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(BatchMatmulAttrs);

bool is_valid(BatchMatmulAttrs const &,
              ParallelTensorShape const &,
              ParallelTensorShape const &);

tl::expected<TensorShape, std::string>
    get_output_shape(BatchMatmulAttrs const &attrs,
                     TensorShape const &input_lhs,
                     TensorShape const &input_rhs);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(BatchMatmulAttrs const &attrs,
                     ParallelTensorShape const &input_lhs,
                     ParallelTensorShape const &input_rhs);
} // namespace FlexFlow

#endif
