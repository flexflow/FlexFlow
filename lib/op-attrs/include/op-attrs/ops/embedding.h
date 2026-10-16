#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_EMBEDDING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_EMBEDDING_H

#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

TensorShape get_output_shape(EmbeddingAttrs const &, TensorShape const &);
TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

ParallelTensorShape get_output_shape(EmbeddingAttrs const &,
                                     ParallelTensorShape const &);
ParallelTensorShape get_weights_shape(EmbeddingAttrs const &,
                                      ParallelTensorShape const &);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/sparse.py#L180-L182
 */
std::map<TensorSlotName, InitializerAttrs> get_initializers(
    EmbeddingAttrs const &,
    std::optional<InitializerAttrs> const &initializer_attrs = std::nullopt);

} // namespace FlexFlow

#endif
