#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_IS_VALID_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_IS_VALID_H

#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"

namespace FlexFlow {

template <typename T>
bool is_valid(T const &t, std::vector<ParallelTensorShape> const &shapes) {
  auto num_outputs = get_num_outputs(t);
  if (num_outputs.has_value() && shapes.size() != num_outputs.value()) {
    return false;
  }

  for (ParallelTensorShape const &shape : shapes) {
    if (!is_valid(shape)) {
      return false;
    }
  }

  return is_valid_internal(t, shapes);
}

bool is_valid_internal(MultiHeadAttentionAttrs const &,
                       std::vector<ParallelTensorShape> const &);
bool is_valid_internal(BatchMatmulAttrs const &,
                       ParallelTensorShape const &,
                       ParallelTensorShape const &);
bool is_valid_internal(CastAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(ConcatAttrs const &,
                       std::vector<ParallelTensorShape> const &);
bool is_valid_internal(Conv2DAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(DropoutAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(ElementBinaryAttrs const &,
                       ParallelTensorShape const &,
                       ParallelTensorShape const &);
bool is_valid_internal(ElementUnaryAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(EmbeddingAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(FlatAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(GatherAttrs const &,
                       ParallelTensorShape const &,
                       ParallelTensorShape const &);
bool is_valid_internal(LayerNormAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(LinearAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(Pool2DAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(ReduceAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(ReductionAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(RepartitionAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(ReplicateAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(ReshapeAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(SoftmaxAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(SplitAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(TopKAttrs const &, ParallelTensorShape const &);
bool is_valid_internal(TransposeAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
