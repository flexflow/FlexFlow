#ifndef _FLEXFLOW_OP_ATTRS_GET_OP_TYPE_H
#define _FLEXFLOW_OP_ATTRS_GET_OP_TYPE_H

#include "operator_attrs.h"
#include "utils/variant.h"

namespace FlexFlow {

OperatorType get_op_type(AggregateAttrs const &);
OperatorType get_op_type(AggregateSpecAttrs const &);
OperatorType get_op_type(BatchMatmulAttrs const &);
OperatorType get_op_type(BatchNormAttrs const &);
OperatorType get_op_type(BroadcastAttrs const &);
OperatorType get_op_type(CastAttrs const &);
OperatorType get_op_type(ConcatAttrs const &);
OperatorType get_op_type(Conv2DAttrs const &);
OperatorType get_op_type(DropoutAttrs const &);
OperatorType get_op_type(ElementBinaryAttrs const &);
OperatorType get_op_type(ElementScalarUnaryAttrs const &);
OperatorType get_op_type(ElementUnaryAttrs const &);
OperatorType get_op_type(EmbeddingAttrs const &);
OperatorType get_op_type(FlatAttrs const &);
OperatorType get_op_type(GatherAttrs const &);
OperatorType get_op_type(Group_byAttrs const &);
OperatorType get_op_type(InputAttrs const &);
OperatorType get_op_type(LayerNormAttrs const &);
OperatorType get_op_type(LinearAttrs const &);
OperatorType get_op_type(MultiHeadAttentionAttrs const &);
OperatorType get_op_type(NoopAttrs const &);
OperatorType get_op_type(Pool2DAttrs const &);
OperatorType get_op_type(ReduceAttrs const &);
OperatorType get_op_type(ReshapeAttrs const &);
OperatorType get_op_type(ReverseAttrs const &);
OperatorType get_op_type(SplitAttrs const &);
OperatorType get_op_type(SoftmaxAttrs const &);
OperatorType get_op_type(TopKAttrs const &);
OperatorType get_op_type(TransposeAttrs const &);
OperatorType get_op_type(CombineAttrs const &);
OperatorType get_op_type(ReductionAttrs const &);
OperatorType get_op_type(RepartitionAttrs const &);
OperatorType get_op_type(ReplicateAttrs const &);

struct GetOpTypeFunctor {
  template <typename T>
  OperatorType operator()(T const &t) {
    return get_op_type(t);
  }
};

template <typename... Ts>
OperatorType get_op_type(variant<Ts...> const &attrs) {
  return visit(GetOpTypeFunctor{}, attrs);
}

} // namespace FlexFlow

#endif
