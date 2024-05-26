#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "core.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(EmbeddingAttrs);

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

ParallelTensorShape get_output_shape(EmbeddingAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
