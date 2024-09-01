#ifndef _FLEXFLOW_OP_ATTRS_GET_OP_TYPE_H
#define _FLEXFLOW_OP_ATTRS_GET_OP_TYPE_H

#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/ops/batch_matmul.dtg.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/broadcast_attrs.dtg.h"
#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/ops/gather_attrs.dtg.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/ops/noop_attrs.dtg.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/ops/split_attrs.dtg.h"
#include "op-attrs/ops/topk_attrs.dtg.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"
#include "op-attrs/ops/weight_attrs.dtg.h"

namespace FlexFlow {

OperatorType get_op_type(BatchMatmulAttrs const &);
OperatorType get_op_type(BatchNormAttrs const &);
OperatorType get_op_type(BroadcastAttrs const &);
OperatorType get_op_type(CastAttrs const &);
OperatorType get_op_type(ConcatAttrs const &);
OperatorType get_op_type(Conv2DAttrs const &);
OperatorType get_op_type(DropoutAttrs const &);
OperatorType get_op_type(ElementBinaryAttrs const &);
OperatorType get_op_type(ElementUnaryAttrs const &);
OperatorType get_op_type(EmbeddingAttrs const &);
OperatorType get_op_type(FlatAttrs const &);
OperatorType get_op_type(GatherAttrs const &);
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
OperatorType get_op_type(WeightAttrs const &);

OperatorType get_op_type(CombineAttrs const &);
OperatorType get_op_type(ReductionAttrs const &);
OperatorType get_op_type(RepartitionAttrs const &);
OperatorType get_op_type(ReplicateAttrs const &);

} // namespace FlexFlow

#endif
