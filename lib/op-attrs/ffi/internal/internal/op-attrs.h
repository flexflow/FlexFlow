#ifndef _FLEXFLOW_OPATTRS_FFI_INTERNAL_INTERNAL_OPATTRS_H
#define _FLEXFLOW_OPATTRS_FFI_INTERNAL_INTERNAL_OPATTRS_H

#include "flexflow/op-attrs.h"
#include "flexflow/utils.h"
#include "internal/opaque.h"
#include "op-attrs/activation.h"
#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/aggregate_spec.h"
#include "op-attrs/ops/aggreagate.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/ops/conv2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/group_by.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/loss_function.h"
#include "op-attrs/ops/noop.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/reduce.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/reshape.h"
#include "op-attrs/ops/reverse.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/split.h"
#include "op-attrs/ops/topk.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/param_sync.h"
#include "op-attrs/dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"

using namespace FlexFlow;

REGISTER_OPAQUE(flexflow_regularizer_attrs_t, optional<RegularizerAttrs>);
REGISTER_OPAQUE(flexflow_ff_dim_t, ff_dim_t);
//REGISTER_OPAQUE(flexflow_dim_ordered_t, DimOrdered);
REGISTER_OPAQUE(flexflow_parallel_dim_t, ParallelDim);
REGISTER_OPAQUE(flexflow_parallel_tensor_dims_t, ParallelTensorDims);
REGISTER_OPAQUE(flexflow_parallel_tensor_shape_t, ParallelTensorShape);
REGISTER_OPAQUE(flexflow_tensor_shape_t, TensorShape);
REGISTER_OPAQUE(flexflow_parallel_tesor_shape_list_t, std::vector<ParallelTensorShape>)
REGISTER_OPAQUE(flexflow_tensor_shape_list_t, std::vector<TensorShape>)

//ops 
REGISTER_OPAQUE(flexflow_aggregate_specattrs_t, AggregateSpecAttrs);
REGISTER_OPAQUE(flexflow_aggregate_attrs_t, AggregateAttrs);
REGISTER_OPAQUE(flexflow_multihead_attention_attrs_t, MultiHeadAttentionAttrs);
REGISTER_OPAQUE(flexflow_multihead_attention_inputs_t, MultiHeadAttentionInputs);
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

flexflow_error_t flexflow_get_output_shape(flexflow_aggregate_specattrs_t, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_t *out, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_list_t);
flexflow_error_t flexflow_get_output_shape(flexflow_aggregate_attrs_t, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_t *out, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_t, flexflow_parallel_tensor_shape_list_t);
//TODO(Note(lambda)):how to support op-attrs/include/op-attrs/ops/attention.h? how to define MultiHeadAttentionInputs<ParallelTensorShape>
flexflow_error_t flexflow_get_output_shape(flexflow_batchnorm_attrs_t, flexflow_parallel_tensor_shape_t * out);//ParallelTensorShape get_output_shape(BatchNormAttrs const &);
flexflow_error_t flexflow_get_kernel_shape(flexflow_conv2d_attrs_t, flexflow_tensor_shape_t * out, flexflow_tensor_shape_t);//TensorShape get_kernel_shape(Conv2DAttrs const &. TensorShape const &);
flexflow_error_t flexflow_get_bias_shape(flexflow_conv2d_attrs_t, flexflow_tensor_shape_t * out, flexflow_tensor_shape_t);//TensorShape get_bias_shape(Conv2DAttrs const &, TensorShape const &);
flexflow_error_t flexflow_get_weights_shape(flexflow_embedding_attrs_t, flexflow_tensor_shape_t * out, flexflow_tensor_shape);//TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);
//how to represent the LossFunction parse_loss_function_name(std::string const &);? because its parameter is std::string 
flexflow_error_t flexflow_get_loss_function(flexflow_other_loss_attrs_t, flexflow_loss_function_t * out);//LossFunction get_loss_function(OtherLossAttrs const &);
flexflow_error_t flexflow_get_loss_function(flexflow_sparse_categorical_crossentropy_loss_attrs_t, flexflow_loss_function_t * out);//LossFunction get_loss_function(SparseCategoricalCrossEntropyLossAttrs const &);
//flexflow_error_t flexflow_get_loss_function(flexflow_loss_)
//TODO(Note lambda):how to define  nner_to_outer_idxs, outer_to_inner_idxs, outer_to_inner(op-attrs/include/op-attrs/dim_ordered.h)
/*
how to define the function 1)its parameter is bool, std::string, or 2)its return type is bool, std::string,int 
 bool is_parallel_op(PCGOperatorAttrs const &);
bool is_valid(PCGOperatorAttrs const &,
              std::vector<ParallelTensorShape> const &);
*/
flexflow_error_t flexflow_get_piece_dims(flexflow_parallel_tensor_dims_t, flexflow_tensor_dims_t * out);//TensorDims get_piece_dims(ParallelTensorDims const &);
flexflow_error_t flexflow_get_tensor_dims_unsafe(flexflow_parallel_tensor_dims_t, flexflow_tensor_dims_t * out);//TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &);
flexflow_error_t flexflow_get_piece_shape(flexflow_parallel_tensor_shape_t, flexflow_tensor_shape_t * out);//TensorShape get_piece_shape(ParallelTensorShape const &);
flexflow_error_t flexflow_get_tensor_shape_unsafe(flexflow_parallel_tensor_shape_t, flexflow_tensor_shape_t * out);//TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &);
flexflow_error_t flexflow_get_tensor_shape_unsafe(flexflow_parallel_tesor_shape_list_t, flexflow_tensor_shape_list_t *out);//std::vector<TensorShape> get_tensor_shape_unsafe(std::vector<ParallelTensorShape> const &);


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
