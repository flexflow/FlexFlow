#include "op-attrs/get_op_type.h"

namespace FlexFlow {

OperatorType get_op_type(AggregateAttrs const &) { return OP_AGGREGATE; }
OperatorType get_op_type(AggregateSpecAttrs const &) { return OP_AGG_SPEC; }
OperatorType get_op_type(BatchMatmulAttrs const &) { return OP_BATCHMATMUL; }
OperatorType get_op_type(CastAttrs const &) { return OP_CAST; }
OperatorType get_op_type(ConcatAttrs const &) { return OP_CONCAT; }
OperatorType get_op_type(Conv2DAttrs const &) { return OP_CONV2D; }
OperatorType get_op_type(DropoutAttrs const &) { return OP_DROPOUT; }
OperatorType get_op_type(ElementBinaryAttrs const &attrs) { return attrs.type; }
OperatorType get_op_type(ElementUnaryAttrs const &attrs) { return attrs.op; }
OperatorType get_op_type(EmbeddingAttrs const &) { return OP_EMBEDDING; }
OperatorType get_op_type(FlatAttrs const &) { return OP_FLAT; }
OperatorType get_op_type(GatherAttrs const &) { return OP_GATHER; }
OperatorType get_op_type(Group_byAttrs const &) { return OP_GROUP_BY; }
OperatorType get_op_type(LayerNormAttrs const &) { return OP_LAYERNORM; }
OperatorType get_op_type(LinearAttrs const &) { return OP_LINEAR; }
OperatorType get_op_type(MultiHeadAttentionAttrs const &) { return OP_MULTIHEAD_ATTENTION; }
OperatorType get_op_type(Pool2DAttrs const &) { return OP_POOL2D; }
OperatorType get_op_type(ReduceAttrs const &) { return OP_REDUCE_SUM; }
OperatorType get_op_type(ReshapeAttrs const &) { return OP_RESHAPE; }
OperatorType get_op_type(SplitAttrs const &) { return OP_SPLIT; }
OperatorType get_op_type(SoftmaxAttrs const &) { return OP_SOFTMAX; }
OperatorType get_op_type(TopKAttrs const &) { return OP_TOPK; }
OperatorType get_op_type(TransposeAttrs const &) { return OP_TRANSPOSE; }
OperatorType get_op_type(CombineAttrs const &) { return OP_COMBINE; }
OperatorType get_op_type(ReductionAttrs const &) { return OP_REDUCTION; }
OperatorType get_op_type(RepartitionAttrs const &) { return OP_REPARTITION; }
OperatorType get_op_type(ReplicateAttrs const &) { return OP_REPLICATE; }
OperatorType get_op_type(FusedParallelOpAttrs const &) { return OP_FUSED_PARALLEL; }

}
