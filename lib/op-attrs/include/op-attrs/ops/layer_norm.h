#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LAYER_NORM_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LAYER_NORM_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &);

TensorShape get_output_shape(LayerNormAttrs const &, TensorShape const &);
TensorShape get_gamma_weights_shape(LayerNormAttrs const &,
                                    TensorShape const &);
TensorShape get_beta_weights_shape(LayerNormAttrs const &, TensorShape const &);

std::map<TensorSlotName, TensorShape>
    get_weight_shapes(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape);

ParallelTensorShape get_output_shape(LayerNormAttrs const &,
                                     ParallelTensorShape const &);
ParallelTensorShape get_gamma_weights_shape(LayerNormAttrs const &,
                                            ParallelTensorShape const &);
ParallelTensorShape get_beta_weights_shape(LayerNormAttrs const &,
                                           ParallelTensorShape const &);

std::map<TensorSlotName, ParallelTensorShape>
    get_weight_shapes(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/normalization.py#L210-L214
 */
std::map<TensorSlotName, InitializerAttrs>
    get_initializers(LayerNormAttrs const &attrs);

} // namespace FlexFlow

#endif
