#ifndef _FLEXFLOW_OPATTRS_FFI_INTERNAL_INTERNAL_OPATTRS_H
#define _FLEXFLOW_OPATTRS_FFI_INTERNAL_INTERNAL_OPATTRS_H

#include "flexflow/op-attrs.h"
#include "internal/opaque.h"
#include "op-attrs/activation.h"
#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/param_sync.h"

using namespace FlexFlow;

REGISTER_OPAQUE(flexflow_regularizer_attrs_t, optional<RegularizerAttrs>);
REGISTER_OPAQUE(flexflow_ff_dim_t, ff_dim_t);
REGISTER_OPAQUE(flexflow_dim_ordered_t, DimOrdered);
REGISTER_OPAQUE(flexflow_parallel_dim_t, ParallelDim);
REGISTER_OPAQUE(flexflow_parallel_tensor_dims_t, ParallelTensorDims);
REGISTER_OPAQUE(flexflow_parallel_tensor_shape_t, ParallelTensorShape);
REGISTER_OPAQUE(flexflow_tensor_shape_t, TensorShape);

//ops 
REGISTER_OPAQUE(flexflow_aggregae_specattrs_t, AggregateSpecAttrs);
REGISTER_OPAQUE(flexflow_aggregate_t, Aggregate);
REGISTER_OPAQUE(flexflow_multihead_attentionattrs_t, MultiHeadAttentionAttrs);
REGISTER_OPAQUE(flexflow_multihead_attentioninputs_t, ultiHeadAttentionInputs);
REGISTER_OPAQUE(flexflow_batchmatmul_attrs_t, BatchMatmulAttrs);
REGISTER_OPAQUE(flexflow_batchnorm_attrs_t, BatchNormAttrs);
REGISTER_OPAQUE(flexflow_broadcast_attrs_t, BroadcastAttrs);
REGISTER_OPAQUE(flexflow_cast_attrs_t, CastAttrs);
REGISTER_OPAQUE(flexflow_combine_attrs_t, CombineAttrs);
REGISTER_OPAQUE(flexflow_concat_attrs_t, ConcatAttrs);
REGISTER_OPAQUE(flexflow_conv2d_attrs_t, Conv2DAttrs);
REGISTER_OPAQUE(flexflow_dropout_attrs_t, DropoutAttrs);
REGISTER_OPAQUE(flexflow_element_sclar_unary_attrs_t, ElementScalarUnaryAttrs);
REGISTER_OPAQUE(flexflow_element_unary_attrs_t, ElementUnaryAttrs);
REGISTER_OPAQUE(flexflow_embedding_attrs_t, EmbeddingAttrs);
REGISTER_OPAQUE(flexflow_flat_attrs_t, FlatAttrs);
REGISTER_OPAQUE(flexflow_gather_attrs_t, GatherAttrs);
REGISTER_OPAQUE(flexflow_group_by_attrs_t, GroupByAttrs);
REGISTER_OPAQUE(flexflow_input_attrs_t, InputAttrs);
REGISTER_OPAQUE(flexflow_layernorm_attrs_t, LayerNormAttrs);
REGISTER_OPAQUE(flexflow_l1_regularizer_attrs_t, L1RegularizerAttrs);
REGISTER_OPAQUE(flexflow_l2_regularizer_attrs_t, L2RegularizerAttrs);
REGISTER_OPAQUE(flexflow_linear_attrs_t, LinearAttrs);
REGISTER_OPAQUE(flexflow_sparse_categorical_crossentropy_loss_attrs_t, SparseCategoricalCrossEntropyLossAttrs);
REGISTER_OPAQUE(flexflow_other_loss_attrs_t, OtherLossAttrs);
REGISTER_OPAQUE(flexflow_noop_attrs_t, NoopAttrs);
REGISTER_OPAQUE(flexflow_pool2d_attrs_t, Pool2DAttrs);
REGISTER_OPAQUE(flexflow_reduce_attrs_t, ReduceAttrs);
REGISTER_OPAQUE(flexflow_reduction_attrs_t, ReductionAttrs);
REGISTER_OPAQUE(flexflow_repartition_attrs_t, RepartitionAttrs);
REGISTER_OPAQUE(flexflow_replicate_attrs_t, ReplicateAttrs);
REGISTER_OPAQUE(flexflow_reshape_attrs_t, ReshapeAttrs);
REGISTER_OPAQUE(flexflow_reverse_attrs_t, ReverseAttrs);
REGISTER_OPAQUE(flexflow_softmax_attrs_t, SoftmaxAttrs);
REGISTER_OPAQUE(flexflow_split_attrs_t, SplitAttrs);
REGISTER_OPAQUE(flexflow_topk_attrs_t, TopKAttrs);

optional<ParamSync> to_internal(flexflow_param_sync_t);
flexflow_param_sync_t to_external(optional<ParamSync>);

DataType to_internal(flexflow_datatype_t);
flexflow_datatype_t to_external(DataType);

optional<Activation> to_internal(flexflow_activation_t);
flexflow_activation_t to_external(optional<Activation>);

PoolOp to_internal(flexflow_pool_op_t e);
flexflow_pool_op_t to_external(PoolOp i);

AggregateOp to_internal(flexflow_aggregate_op_t e);
flexflow_aggregate_op_t to_external(AggregateOp i);

OperatorType to_internal(flexflow_op_type_t e);
flexflow_op_type_t to_external(OperatorType i);

#endif
