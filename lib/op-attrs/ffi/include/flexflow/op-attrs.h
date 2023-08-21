#ifndef _FLEXFLOW_OPATTRS_FFI_INCLUDE_FLEXFLOW_OPATTRS_H
#define _FLEXFLOW_OPATTRS_FFI_INCLUDE_FLEXFLOW_OPATTRS_H

#include "flexflow/utils.h"

FLEXFLOW_FFI_BEGIN()

// Error handling

typedef enum {
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_PARAM_SYNC_VALUE,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_DATATYPE_VALUE,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_ACTIVATION_VALUE,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_POOL_OP_VALUE,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_AGGREGATE_OP_VALUE,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_OP_TYPE_VALUE,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_LOSS_FUNCTION_VALUE,
} flexflow_opattrs_error_code_t;

FF_NEW_OPAQUE_TYPE(flexflow_opattrs_error_t);
flexflow_error_t flexflow_opattrs_error_wrap(flexflow_opattrs_error_t);
flexflow_error_t flexflow_opattrs_error_unwrap(flexflow_error_t,
                                               flexflow_opattrs_error_t *);
flexflow_error_t flexflow_opattrs_error_is_ok(flexflow_opattrs_error_t, bool *);
flexflow_error_t flexflow_opattrs_error_get_string(flexflow_opattrs_error_t,
                                                   char **);
flexflow_error_t
    flexflow_opattrs_error_get_error_code(flexflow_opattrs_error_t,
                                          flexflow_opattrs_error_code_t *);
flexflow_error_t flexflow_opattrs_error_destroy(flexflow_opattrs_error_t);

//

FF_NEW_OPAQUE_TYPE(flexflow_regularizer_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_dim_ordered_t);
FF_NEW_OPAQUE_TYPE(flexflow_ff_dim_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_dim_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_tensor_dims_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_tensor_shape_t);
FF_NEW_OPAQUE_TYPE(flexflow_tensor_shape_t);
FF_NEW_OPAQUE_TYPE(
    flexflow_parallel_tesor_shape_list_t); // std::vector<ParallelTensorShape>
FF_NEW_OPAQUE_TYPE(flexflow_tensor_shape_list_t); // std::vector<TensorShape>

// ops
FF_NEW_OPAQUE_TYPE(flexflow_aggregate_specattrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_aggregate_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_multihead_attention_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_multihead_attention_inputs_parallel_tensor_shape_t);
FF_NEW_OPAQUE_TYPE(flexflow_multihead_attention_inputs_tensor_shape_t);
FF_NEW_OPAQUE_TYPE(flexflow_batchmatmul_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_batchnorm_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_broadcast_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_cast_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_combine_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_concat_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_conv2d_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_dropout_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_element_sclar_unary_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_element_unary_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_embedding_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_flat_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_gather_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_group_by_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_input_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_layernorm_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_l1_regularizer_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_l2_regularizer_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_linear_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_sparse_categorical_crossentropy_loss_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_other_loss_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_loss_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_noop_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_pool2d_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_reduce_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_reduction_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_repartition_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_replicate_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_reshape_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_reverse_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_softmax_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_split_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_topk_attrs_t);
FF_NEW_OPAQUE_TYPE(flexflow_transpose_attrs_t);

typedef enum {
  FLEXFLOW_DATATYPE_BOOL,
  FLEXFLOW_DATATYPE_INT32,
  FLEXFLOW_DATATYPE_INT64,
  FLEXFLOW_DATATYPE_HALF,
  FLEXFLOW_DATATYPE_FLOAT,
  FLEXFLOW_DATATYPE_DOUBLE
} flexflow_datatype_t;

typedef enum {
  FLEXFLOW_ACTIVATION_RELU,
  FLEXFLOW_ACTIVATION_SIGMOID,
  FLEXFLOW_ACTIVATION_TANH,
  FLEXFLOW_ACTIVATION_GELU,
  FLEXFLOW_ACTIVATION_NONE
} flexflow_activation_t;

typedef enum {
  FLEXFLOW_POOL_OP_MAX,
  FLEXFLOW_POOL_OP_AVG,
} flexflow_pool_op_t;

typedef enum {
  FLEXFLOW_PARAM_SYNC_PARAMETER_SERVER,
  FLEXFLOW_PARAM_SYNC_NCCL,
  FLEXFLOW_PARAM_SYNC_NONE
} flexflow_param_sync_t;

typedef enum {
  FLEXFLOW_AGGREGATE_OP_SUM,
  FLEXFLOW_AGGREGATE_OP_AVG,
} flexflow_aggregate_op_t;

typedef enum {
  FLEXFLOW_LOSS_FUNCTION_CATEGORICAL_CROSSENTROPY,
  FLEXFLOW_LOSS_FUNCTION_SPARSE_CATEGORICAL_CROSSENTROPY,
  FLEXFLOW_LOSS_FUNCTION_MEAN_SQUARED_ERROR_AVG_REDUCE,
  FLEXFLOW_LOSS_FUNCTION_MEAN_SQUARED_ERROR_SUM_REDUCE,
  FLEXFLOW_LOSS_FUNCTION_IDENTITY,
} flexflow_loss_function_t;

typedef enum { // does _not_ have to stay synchronized with op-attrs/op.h
  FLEXFLOW_OP_TYPE_NOOP,
  FLEXFLOW_OP_TYPE_INPUT,
  FLEXFLOW_OP_TYPE_WEIGHT,
  FLEXFLOW_OP_TYPE_CONV2D,
  FLEXFLOW_OP_TYPE_DROPOUT,
  FLEXFLOW_OP_TYPE_LINEAR,
  FLEXFLOW_OP_TYPE_BATCHMATMUL,
  FLEXFLOW_OP_TYPE_POOL2D,
  FLEXFLOW_OP_TYPE_SCALAR_MULTIPLY,
  FLEXFLOW_OP_TYPE_SCALAR_ADD,
  FLEXFLOW_OP_TYPE_SCALAR_FLOOR_DIV,
  FLEXFLOW_OP_TYPE_SCALAR_TRUE_DIV,
  FLEXFLOW_OP_TYPE_SCALAR_SUB,
  FLEXFLOW_OP_TYPE_RELU,
  FLEXFLOW_OP_TYPE_IDENTITY,
  FLEXFLOW_OP_TYPE_SIGMOID,
  FLEXFLOW_OP_TYPE_TANH,
  FLEXFLOW_OP_TYPE_ELU,
  FLEXFLOW_OP_TYPE_FLAT,
  FLEXFLOW_OP_TYPE_SOFTMAX,
  FLEXFLOW_OP_TYPE_BATCHNORM,
  FLEXFLOW_OP_TYPE_CONCAT,
  FLEXFLOW_OP_TYPE_SPLIT,
  FLEXFLOW_OP_TYPE_EMBEDDING,
  FLEXFLOW_OP_TYPE_GROUP_BY,
  FLEXFLOW_OP_TYPE_CACHE,
  FLEXFLOW_OP_TYPE_AGGREGATE,
  FLEXFLOW_OP_TYPE_AGG_SPEC,
  FLEXFLOW_OP_TYPE_RESHAPE,
  FLEXFLOW_OP_TYPE_REVERSE,
  FLEXFLOW_OP_TYPE_TRANSPOSE,
  FLEXFLOW_OP_TYPE_EW_ADD,
  FLEXFLOW_OP_TYPE_EW_MUL,
  FLEXFLOW_OP_TYPE_MATMUL,
  FLEXFLOW_OP_TYPE_MUL,
  FLEXFLOW_OP_TYPE_ENLARGE,
  FLEXFLOW_OP_TYPE_SQUEEZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  FLEXFLOW_OP_TYPE_UNSQUEEZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  FLEXFLOW_OP_TYPE_EW_SUB, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  FLEXFLOW_OP_TYPE_EW_DIV, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  FLEXFLOW_OP_TYPE_EW_EQUAL, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  FLEXFLOW_OP_TYPE_EW_GREATER, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  FLEXFLOW_OP_TYPE_EW_LESS, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  FLEXFLOW_OP_TYPE_EW_MAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  FLEXFLOW_OP_TYPE_EW_MIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  FLEXFLOW_OP_TYPE_REDUCE_ARGMAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  FLEXFLOW_OP_TYPE_REDUCE_ARGMIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  FLEXFLOW_OP_TYPE_REDUCE_MAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  FLEXFLOW_OP_TYPE_REDUCE_MEAN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  FLEXFLOW_OP_TYPE_REDUCE_MIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  FLEXFLOW_OP_TYPE_REDUCE_PROD, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  FLEXFLOW_OP_TYPE_REDUCE_SUM, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  FLEXFLOW_OP_TYPE_PAD, // https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  FLEXFLOW_OP_TYPE_SHAPE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  FLEXFLOW_OP_TYPE_SIZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  FLEXFLOW_OP_TYPE_TOPK, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  FLEXFLOW_OP_TYPE_WHERE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  FLEXFLOW_OP_TYPE_CEIL, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  FLEXFLOW_OP_TYPE_CAST, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  FLEXFLOW_OP_TYPE_EXP, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  FLEXFLOW_OP_TYPE_ROUND, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  FLEXFLOW_OP_TYPE_LOG, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  FLEXFLOW_OP_TYPE_LOGICAL_NOT, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  FLEXFLOW_OP_TYPE_SQRT, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  FLEXFLOW_OP_TYPE_SIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin
  FLEXFLOW_OP_TYPE_COS, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos
  FLEXFLOW_OP_TYPE_LEAKYRELU,
  FLEXFLOW_OP_TYPE_SLICE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  FLEXFLOW_OP_TYPE_RESIZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  FLEXFLOW_OP_TYPE_PRELU, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
  FLEXFLOW_OP_TYPE_GELU,
  FLEXFLOW_OP_TYPE_MULTIHEAD_ATTENTION,
  FLEXFLOW_OP_TYPE_FUSED, // Fused operator type for internal fusion
                          // optimizations
  FLEXFLOW_OP_TYPE_RSQRT, // https://pytorch.org/docs/stable/generated/torch.rsqrt.html
  FLEXFLOW_OP_TYPE_POW, // https://pytorch.org/docs/stable/generated/torch.pow.html
  FLEXFLOW_OP_TYPE_MEAN, // https://pytorch.org/docs/stable/generated/torch.mean.html
  FLEXFLOW_OP_TYPE_LAYERNORM,
  FLEXFLOW_OP_TYPE_GATHER, // https://pytorch.org/docs/stable/generated/torch.gather.html
  FLEXFLOW_OP_TYPE_BROADCAST,
  FLEXFLOW_OP_TYPE_REPARTITION,
  FLEXFLOW_OP_TYPE_COMBINE,
  FLEXFLOW_OP_TYPE_REPLICATE,
  FLEXFLOW_OP_TYPE_REDUCTION,
  FLEXFLOW_OP_TYPE_BATCH,
  FLEXFLOW_OP_TYPE_PIPELINE,
  FLEXFLOW_OP_TYPE_FUSED_PARALLEL,
} flexflow_op_type_t;

typedef struct {
  flexflow_op_type_t op_type;
  void *data;
} flexflow_operator_attrs;

FF_NEW_OPAQUE_TYPE(flexflow_operator_attrs_t);

flexflow_opattrs_error_t flexflow_get_datatype_size(flexflow_datatype_t,
                                                    int *out);
flexflow_opattrs_error_t
    flexflow_operator_attrs_get_op_type(flexflow_operator_attrs_t,
                                        flexflow_op_type_t *out);

FLEXFLOW_FFI_END()

#endif
