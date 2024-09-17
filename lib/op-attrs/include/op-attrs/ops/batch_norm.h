#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_batch_norm_incoming_tensor_roles(BatchNormAttrs const &);

tl::expected<TensorShape, std::string> get_output_shape(BatchNormAttrs const &,
                                                        TensorShape const &);
tl::expected<TensorShape, std::string>
    get_gamma_weights_shape(BatchNormAttrs const &, TensorShape const &);
tl::expected<TensorShape, std::string>
    get_beta_weights_shape(BatchNormAttrs const &, TensorShape const &);

tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(BatchNormAttrs const &,
                                    ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_gamma_weights_parallel_dim_degrees(BatchNormAttrs const &,
                                           ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_beta_weights_parallel_dim_degrees(BatchNormAttrs const &,
                                          ParallelTensorDimDegrees const &);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(BatchNormAttrs const &, ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(BatchNormAttrs const &,
                            ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(BatchNormAttrs const &, ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(BatchNormAttrs);

} // namespace FlexFlow

#endif
