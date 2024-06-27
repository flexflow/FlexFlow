#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(EmbeddingAttrs);

tl::expected<TensorShape, std::string> get_output_shape(EmbeddingAttrs const &,
                                                        TensorShape const &);
tl::expected<TensorShape, std::string> get_weights_shape(EmbeddingAttrs const &,
                                                         TensorShape const &);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(EmbeddingAttrs const &, ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(EmbeddingAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
