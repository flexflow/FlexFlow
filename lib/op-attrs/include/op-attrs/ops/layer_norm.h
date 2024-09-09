#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

std::vector<IncomingTensorRole> get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &);

tl::expected<TensorShape, std::string> get_output_shape(LayerNormAttrs const &,
                                                        TensorShape const &);
tl::expected<TensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &, TensorShape const &);
tl::expected<TensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &, TensorShape const &);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LayerNormAttrs const &, ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &,
                            ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &, ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(LayerNormAttrs);

} // namespace FlexFlow

#endif
