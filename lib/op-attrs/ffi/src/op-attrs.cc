#include "flexflow/op-attrs.h"
#include "flexflow/utils.h"
#include "internal/enums.h"
#include "internal/error.h"
#include "internal/op-attrs.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/loss_functions.h"
#include "utils/bidict.h"
#include "utils/exception.h"

flexflow_utils_exception_t
    make_opattrs_exception(flexflow_opattrs_error_code_t);

flexflow_error_t flexflow_opattrs_error_wrap(flexflow_opattrs_error_t e) {
  return flexflow_error_wrap(FLEXFLOW_ERROR_SOURCE_OPATTRS, *unwrap_opaque(e));
}

flexflow_error_t flexflow_opattrs_error_unwrap(flexflow_error_t err,
                                               flexflow_opattrs_error_t *out) {
  return flexflow_error_unwrap(err, FLEXFLOW_ERROR_SOURCE_OPATTRS, out);
}

flexflow_error_t flexflow_opattrs_error_is_ok(flexflow_opattrs_error_t err,
                                              bool *out) {
  *out = false;
  return status_ok();
}

flexflow_error_t flexflow_opattrs_error_get_string(flexflow_opattrs_error_t err,
                                                   char **m_out) {
  flexflow_opattrs_error_code_t err_code;
  flexflow_opattrs_error_get_error_code(err, &err_code);
  auto out = const_cast<char const **>(m_out);
  switch (err_code) {
    case FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_PARAM_SYNC_VALUE:
      *out = strdup("Invalid param sync value");
      break;
    case FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_DATATYPE_VALUE:
      *out = strdup("Invalid datatype value");
      break;
    case FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_ACTIVATION_VALUE:
      *out = strdup("Invalid activation value");
      break;
    case FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_POOL_OP_VALUE:
      *out = strdup("Invalid pool op value");
      break;
    case FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_AGGREGATE_OP_VALUE:
      *out = strdup("Invalid aggregate op value");
      break;
    case FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_OP_TYPE_VALUE:
      *out = strdup("Invalid op type value");
      break;
    default:
      *out = strdup("Unknown error");
  }
  return status_ok();
}

flexflow_error_t
    flexflow_opattrs_error_get_error_code(flexflow_opattrs_error_t err,
                                          flexflow_opattrs_error_code_t *out) {
  flexflow_opattrs_error_t opaque;
  RAISE_FLEXFLOW(flexflow_opattrs_error_unwrap(err, &opaque));
  interal_flexflow_opattrs_error_t const *unwrapped = unwrap_opaque(opaque);
  *out = unwrapped->code;
  return status_ok();
}

flexflow_error_t flexflow_opattrs_error_destroy(flexflow_opattrs_error_t err) {
  return status_ok(); //  Note(lambda): this is follow the
                      //  https://github.com/lockshaw/FlexFlow/blob/expanded-ffi/lib/pcg/ffi/src/pcg.cc#L71-#L72
}

REGISTER_FFI_ENUM(flexflow_param_sync_t,
                  ParamSync,
                  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_PARAM_SYNC_VALUE,
                  {{FLEXFLOW_PARAM_SYNC_PARAMETER_SERVER, ParamSync::PS},
                   {FLEXFLOW_PARAM_SYNC_NCCL, ParamSync::NCCL}});

REGISTER_FFI_ENUM(flexflow_datatype_t,
                  DataType,
                  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_DATATYPE_VALUE,
                  {{FLEXFLOW_DATATYPE_BOOL, DataType::BOOL},
                   {FLEXFLOW_DATATYPE_INT32, DataType::INT32},
                   {FLEXFLOW_DATATYPE_INT64, DataType::INT64},
                   {FLEXFLOW_DATATYPE_HALF, DataType::HALF},
                   {FLEXFLOW_DATATYPE_FLOAT, DataType::FLOAT},
                   {FLEXFLOW_DATATYPE_DOUBLE, DataType::DOUBLE}});

REGISTER_FFI_ENUM(flexflow_activation_t,
                  optional<Activation>,
                  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_ACTIVATION_VALUE,
                  {{FLEXFLOW_ACTIVATION_RELU, Activation::RELU},
                   {FLEXFLOW_ACTIVATION_SIGMOID, Activation::SIGMOID},
                   {FLEXFLOW_ACTIVATION_TANH, Activation::TANH},
                   {FLEXFLOW_ACTIVATION_GELU, Activation::GELU},
                   {FLEXFLOW_ACTIVATION_NONE, nullopt}});

REGISTER_FFI_ENUM(flexflow_pool_op_t,
                  PoolOp,
                  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_POOL_OP_VALUE,
                  {{FLEXFLOW_POOL_OP_MAX, PoolOp::MAX},
                   {FLEXFLOW_POOL_OP_AVG, PoolOp::AVG}});

REGISTER_FFI_ENUM(flexflow_aggregate_op_t,
                  AggregateOp,
                  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_AGGREGATE_OP_VALUE,
                  {{FLEXFLOW_AGGREGATE_OP_SUM, AggregateOp::SUM},
                   {FLEXFLOW_AGGREGATE_OP_AVG, AggregateOp::AVG}});

REGISTER_FFI_NUM(flexflow_loss_function_t,
                 LossFunction,
                 FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_LOSS_FUNCTION_VALUE,
                 {{FLEXFLOW_LOSS_FUNCTION_CATEGORICAL_CROSSENTROPY,
                   LossFunction::CATEGORICAL_CROSSENTROPY},
                  {FLEXFLOW_LOSS_FUNCTION_SPARSE_CATEGORICAL_CROSSENTROPY,
                   LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY},
                  {FLEXFLOW_LOSS_FUNCTION_MEAN_SQUARED_ERROR,
                   LossFunction::MEAN_SQUARED_ERROR},
                  {FLEXFLOW_LOSS_FUNCTION_MEAN_ABSOLUTE_ERROR,
                   LossFunction::MEAN_ABSOLUTE_ERROR}});

REGISTER_FFI_ENUM(flexflow_op_type_t,
                  OperatorType,
                  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_OP_TYPE_VALUE,
                  {
                      {FLEXFLOW_OP_TYPE_NOOP, Op::NOOP},
                      {FLEXFLOW_OP_TYPE_INPUT, Op::INPUT},
                      {FLEXFLOW_OP_TYPE_WEIGHT, Op::WEIGHT},
                      {FLEXFLOW_OP_TYPE_CONV2D, Op::CONV2D},
                      {FLEXFLOW_OP_TYPE_DROPOUT, Op::DROPOUT},
                      {FLEXFLOW_OP_TYPE_LINEAR, Op::LINEAR},
                      {FLEXFLOW_OP_TYPE_BATCHMATMUL, Op::BATCHMATMUL},
                      {FLEXFLOW_OP_TYPE_POOL2D, Op::POOL2D},
                      {FLEXFLOW_OP_TYPE_SCALAR_MULTIPLY, Op::SCALAR_MULTIPLY},
                      {FLEXFLOW_OP_TYPE_SCALAR_ADD, Op::SCALAR_ADD},
                      {FLEXFLOW_OP_TYPE_SCALAR_FLOOR_DIV, Op::SCALAR_FLOOR_DIV},
                      {FLEXFLOW_OP_TYPE_SCALAR_TRUE_DIV, Op::SCALAR_TRUE_DIV},
                      {FLEXFLOW_OP_TYPE_SCALAR_SUB, Op::SCALAR_SUB},
                      {FLEXFLOW_OP_TYPE_RELU, Op::RELU},
                      {FLEXFLOW_OP_TYPE_IDENTITY, Op::IDENTITY},
                      {FLEXFLOW_OP_TYPE_SIGMOID, Op::SIGMOID},
                      {FLEXFLOW_OP_TYPE_TANH, Op::TANH},
                      {FLEXFLOW_OP_TYPE_ELU, Op::ELU},
                      {FLEXFLOW_OP_TYPE_FLAT, Op::FLAT},
                      {FLEXFLOW_OP_TYPE_SOFTMAX, Op::SOFTMAX},
                      {FLEXFLOW_OP_TYPE_BATCHNORM, Op::BATCHNORM},
                      {FLEXFLOW_OP_TYPE_CONCAT, Op::CONCAT},
                      {FLEXFLOW_OP_TYPE_SPLIT, Op::SPLIT},
                      {FLEXFLOW_OP_TYPE_EMBEDDING, Op::EMBEDDING},
                      {FLEXFLOW_OP_TYPE_GROUP_BY, Op::GROUP_BY},
                      {FLEXFLOW_OP_TYPE_CACHE, Op::CACHE},
                      {FLEXFLOW_OP_TYPE_AGGREGATE, Op::AGGREGATE},
                      {FLEXFLOW_OP_TYPE_AGG_SPEC, Op::AGG_SPEC},
                      {FLEXFLOW_OP_TYPE_RESHAPE, Op::RESHAPE},
                      {FLEXFLOW_OP_TYPE_REVERSE, Op::REVERSE},
                      {FLEXFLOW_OP_TYPE_TRANSPOSE, Op::TRANSPOSE},
                      {FLEXFLOW_OP_TYPE_EW_ADD, Op::EW_ADD},
                      {FLEXFLOW_OP_TYPE_EW_MUL, Op::EW_MUL},
                      {FLEXFLOW_OP_TYPE_MATMUL, Op::MATMUL},
                      {FLEXFLOW_OP_TYPE_MUL, Op::MUL},
                      {FLEXFLOW_OP_TYPE_ENLARGE, Op::ENLARGE},
                      {FLEXFLOW_OP_TYPE_SQUEEZE, Op::SQUEEZE},
                      {FLEXFLOW_OP_TYPE_UNSQUEEZE, Op::UNSQUEEZE},
                      {FLEXFLOW_OP_TYPE_EW_SUB, Op::EW_SUB},
                      {FLEXFLOW_OP_TYPE_EW_DIV, Op::EW_DIV},
                      {FLEXFLOW_OP_TYPE_EW_EQUAL, Op::EW_EQUAL},
                      {FLEXFLOW_OP_TYPE_EW_GREATER, Op::EW_GREATER},
                      {FLEXFLOW_OP_TYPE_EW_LESS, Op::EW_LESS},
                      {FLEXFLOW_OP_TYPE_EW_MAX, Op::EW_MAX},
                      {FLEXFLOW_OP_TYPE_EW_MIN, Op::EW_MIN},
                      {FLEXFLOW_OP_TYPE_REDUCE_ARGMAX, Op::REDUCE_ARGMAX},
                      {FLEXFLOW_OP_TYPE_REDUCE_ARGMIN, Op::REDUCE_ARGMIN},
                      {FLEXFLOW_OP_TYPE_REDUCE_MAX, Op::REDUCE_MAX},
                      {FLEXFLOW_OP_TYPE_REDUCE_MEAN, Op::REDUCE_MEAN},
                      {FLEXFLOW_OP_TYPE_REDUCE_MIN, Op::REDUCE_MIN},
                      {FLEXFLOW_OP_TYPE_REDUCE_PROD, Op::REDUCE_PROD},
                      {FLEXFLOW_OP_TYPE_REDUCE_SUM, Op::REDUCE_SUM},
                      {FLEXFLOW_OP_TYPE_PAD, Op::PAD},
                      {FLEXFLOW_OP_TYPE_SHAPE, Op::SHAPE},
                      {FLEXFLOW_OP_TYPE_SIZE, Op::SIZE},
                      {FLEXFLOW_OP_TYPE_TOPK, Op::TOPK},
                      {FLEXFLOW_OP_TYPE_WHERE, Op::WHERE},
                      {FLEXFLOW_OP_TYPE_CEIL, Op::CEIL},
                      {FLEXFLOW_OP_TYPE_CAST, Op::CAST},
                      {FLEXFLOW_OP_TYPE_EXP, Op::EXP},
                      {FLEXFLOW_OP_TYPE_ROUND, Op::ROUND},
                      {FLEXFLOW_OP_TYPE_LOG, Op::LOG},
                      {FLEXFLOW_OP_TYPE_LOGICAL_NOT, Op::LOGICAL_NOT},
                      {FLEXFLOW_OP_TYPE_SQRT, Op::SQRT},
                      {FLEXFLOW_OP_TYPE_SIN, Op::SIN},
                      {FLEXFLOW_OP_TYPE_COS, Op::COS},
                      {FLEXFLOW_OP_TYPE_LEAKYRELU, Op::LEAKYRELU},
                      {FLEXFLOW_OP_TYPE_SLICE, Op::SLICE},
                      {FLEXFLOW_OP_TYPE_RESIZE, Op::RESIZE},
                      {FLEXFLOW_OP_TYPE_PRELU, Op::PRELU},
                      {FLEXFLOW_OP_TYPE_GELU, Op::GELU},
                      {FLEXFLOW_OP_TYPE_MULTIHEAD_ATTENTION,
                       Op::MULTIHEAD_ATTENTION},
                      {FLEXFLOW_OP_TYPE_FUSED, Op::FUSED},
                      {FLEXFLOW_OP_TYPE_RSQRT, Op::RSQRT},
                      {FLEXFLOW_OP_TYPE_POW, Op::POW},
                      {FLEXFLOW_OP_TYPE_MEAN, Op::MEAN},
                      {FLEXFLOW_OP_TYPE_LAYERNORM, Op::LAYERNORM},
                      {FLEXFLOW_OP_TYPE_GATHER, Op::GATHER},
                      {FLEXFLOW_OP_TYPE_BROADCAST, Op::BROADCAST},
                      {FLEXFLOW_OP_TYPE_REPARTITION, Op::REPARTITION},
                      {FLEXFLOW_OP_TYPE_COMBINE, Op::COMBINE},
                      {FLEXFLOW_OP_TYPE_REPLICATE, Op::REPLICATE},
                      {FLEXFLOW_OP_TYPE_REDUCTION, Op::REDUCTION},
                      {FLEXFLOW_OP_TYPE_BATCH, Op::BATCH},
                      {FLEXFLOW_OP_TYPE_PIPELINE, Op::PIPELINE},
                      {FLEXFLOW_OP_TYPE_FUSED_PARALLEL, Op::FUSED_PARALLEL},
                  });

flexflow_error_t make_opattrs_error(flexflow_opattrs_error_code_t);

flexflow_error_t flexflow_get_output_shape(
    flexflow_aggregate_specattrs_t aggregate_spec_attrs,
    flexflow_parallel_tensor_shape_t gate_preds,
    flexflow_parallel_tensor_shape_t *out,
    flexflow_parallel_tensor_shape_t gate_assign,
    flexflow_parallel_tensor_shape_t true_gate_assign,
    flexflow_parallel_tensor_shape_t gate_gridents_full,
    flexflow_parallel_tensor_shape_t *exp_preds,
    int num_exp_preds) {
  return handle_errors(out, [&]) {
    return get_out_shape(deref_opaque(aggregate_spec_attrs),
                         deref_opaque(gate_preds),
                         deref_opaque(gate_assign),
                         deref_opaque(true_gate_assign),
                         deref_opaque(gate_gridents_full),
                         c_deref_opaque_list(exp_preds, num_exp_preds));
  }
}

flexflow_error_t flexflow_is_valid(
    flexflow_aggregate_attrs_t aggregate_attrs,
    flexflow_parallel_tensor_shape_t gate_preds,
    bool *out,
    flexflow_parallel_tensor_shape_t gate_assign,
    flexflow_parallel_tensor_shape_t true_gate_assign,
    flexflow_parallel_tensor_shape_t full_gate_gradients,
    flexflow_parallel_tensor_shape_t *exp_preds int num_exp_preds) {
  return handle_errors(out, [&]) {
    return is_valid(deref_opaque(aggregate_attrs),
                    deref_opaque(gate_preds),
                    deref_opaque(gate_assign),
                    deref_opaque(true_gate_assign),
                    deref_opaque(full_gate_gradients),
                    c_deref_opaque_list(exp_preds, num_exp_preds));
  }
}

flexflow_error_t flexflow_get_output_shape(
    flexflow_aggregate_attrs_t aggregate_attrs,
    flexflow_parallel_tensor_shape_t gate_preds,
    flexflow_parallel_tensor_shape_t *out,
    flexflow_parallel_tensor_shape_t gate_assign,
    flexflow_parallel_tensor_shape_t true_gate_assign,
    flexflow_parallel_tensor_shape_t full_gate_gradients,
    flexflow_parallel_tensor_shape_t *exp_preds,
    int num_exp_preds) {
  return handle_errors(out, [&]) {
    return get_out_shape(deref_opaque(aggregate_attrs),
                         deref_opaque(gate_preds),
                         deref_opaque(gate_assign),
                         deref_opaque(true_gate_assign),
                         deref_opaque(full_gate_gradients),
                         c_deref_opaque_list(exp_preds, num_exp_preds));
  }
}

flexflow_error_t flexflow_get_kProjSize(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs, int *out) {
  return handle_errors(out, [&]) {
    return get_kProjSize(deref_opaque(multi_head_attention_attrs));
  }
}

flexflow_error_t flexflow_get_vProjSize(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs, int *out) {
  return handle_errors(out, [&]) {
    return get_vProjSize(deref_opaque(multi_head_attention_attrs));
  }
}

flexflow_error_t flexflow_get_kProjSize(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs, int *out) {
  return handle_errors(out, [&]) {
    return get_kProjSize(deref_opaque(multi_head_attention_attrs));
  }
}

flexflow_error_t flexflow_get_oProjSize(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs, int *out) {
  return handle_errors(out, [&]) {
    return get_oProjSize(deref_opaque(multi_head_attention_attrs));
  }
}

flexflow_error_t flexflow_get_qSize(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_qSize(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_kSize(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_kSize(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_vSize(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_vSize(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_oSize(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_oSize(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_qoSeqLength(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_qoSeqLength(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_kvSeqLength(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_kvSeqLength(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_num_samples(
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    int *out) {
  return handle_errors(out, [&]) {
    return get_num_samples(deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_weights_shape(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs,
    flexflow_multihead_attention_inputs_tensor_shape_t
        multi_head_attention_inputs,
    flexflow_tensor_shape_t *out) {
  return handle_errors(out, [&]) {
    return get_weights_shape(deref_opaque(multi_head_attention_attrs),
                             deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_weights_shape(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs,
    flexflow_multihead_attention_inputs_parallel_tensor_shape_t
        multi_head_attention_inputs,
    flexflow_parallel_tensor_shape_t *out) {
  return handle_errors(out, [&]) {
    return get_weights_shape(deref_opaque(multi_head_attention_attrs),
                             deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_output_shape(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs,
    flexflow_multihead_attention_inputs_tensor_shape_t
        multi_head_attention_inputs,
    flexflow_parallel_tensor_shape_t *out, ) {
  return handle_errors(out, [&]) {
    return get_output_shape(deref_opaque(multi_head_attention_attrs),
                            deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t flexflow_get_output_shape(
    flexflow_multihead_attention_attrs_t multi_head_attention_attrs,
    flexflow_multihead_attention_inputs_tensor_shape_t
        multi_head_attention_inputs,
    flexflow_tensor_shape_t *out) {
  return handle_errors(out, [&]) {
    return get_output_shape(deref_opaque(multi_head_attention_attrs),
                            deref_opaque(multi_head_attention_inputs));
  }
}

flexflow_error_t
    flexflow_get_output_shape(flexflow_batchnorm_attrs_t batchnorm_attrs,
                              flexflow_parallel_tensor_shape_t *out) {
  return handle_errors(out, [&]) {
    return get_output_shape(deref_opaque(batchnorm_attrs));
  }
}

flexflow_error_t
    flexflow_get_kernel_shape(flexflow_conv2d_attrs_t conv2d_attrs,
                              flexflow_tensor_shape_t *out,
                              flexflow_tensor_shape_t input_shape) {
  return handle_errors(out, [&]) {
    return get_kernel_shape(deref_opaque(conv2d_attrs),
                            deref_opaque(input_shape));
  }
}

flexflow_error_t flexflow_get_bias_shape(flexflow_conv2d_attrs_t conv2d_attrs,
                                         flexflow_tensor_shape_t *out,
                                         flexflow_tensor_shape_t input_shape) {
  return handle_errors(out, [&]) {
    return get_bias_shape(deref_opaque(conv2d_attrs),
                          deref_opaque(input_shape));
  }
}

flexflow_error_t
    flexflow_get_weights_shape(flexflow_embedding_attrs_t embedding_attrs,
                               flexflow_tensor_shape_t *out,
                               flexflow_tensor_shape_t input_shape) {
  return handle_errors(out, [&]) {
    return get_weights_shape(deref_opaque(embedding_attrs),
                             deref_opaque(input_shape));
  }
}

flexflow_error_t
    flexflow_parse_loss_function_name(char **raw_name,
                                      flexflow_loss_function_t *out) {
  NOT_IMPLEMENTED(); // Note(lambda):how to implement the function
}

flexflow_error_t flexflow_is_valid(flexflow_parallel_dim_t parallel_dim_t,
                                   bool *out) {
  return handle_errors(out, [&]) {
    return is_valid(deref_opaque(parallel_dim_t));
  }
}

flexflow_error_t flexflow_is_replica_dim(flexflow_parallel_dim_t parallel_dim_t,
                                         bool *out) {
  return handle_errors(out, [&]) {
    return is_replica_dim(deref_opaque(parallel_dim_t));
  }
}

flexflow_error_t
    flexflow_is_valid(flexflow_parallel_tensor_dims_t parallel_tensor_dims_t,
                      bool *out) {
  return handle_errors(out, [&]) {
    return is_valid(deref_opaque(parallel_tensor_dims_t));
  }
}

flexflow_error_t flexflow_get_piece_dims(
    flexflow_parallel_tensor_dims_t parallel_tensor_dims_t,
    flexflow_tensor_dims_t *out) {
  return handle_errors(out, [&]) {
    return get_piece_dims(deref_opaque(parallel_tensor_dims_t));
  }
}

flexflow_error_t flexflow_get_tensor_dims_unsafe(
    flexflow_parallel_tensor_dims_t tensor_dims_t,
    flexflow_tensor_dims_t *out) {
  return handle_errors(out, [&]) {
    return get_tensor_dims_unsafe(deref_opaque(tensor_dims_t));
  }
}

flexflow_error_t flexflow_get_piece_shape(
    flexflow_parallel_tensor_shape_t parallel_tensor_shape,
    flexflow_tensor_shape_t *out) {
  return handle_errors(out, [&]) {
    return get_piece_shape(deref_opaque(parallel_tensor_shape));
  }
}

flexflow_error_t flexflow_get_num_replica_dims(
    flexflolw_parallel_tensor_shape_t parallel_tensor_shape, int *out) {
  return handle_errors(out, [&]) {
    return get_num_replica_dims(deref_opaque(parallel_tensor_shape));
  }
}

flexflow_error_t flexflow_get_num_replicas(
    flexflow_parallel_tensor_shape_t parallel_tensor_shape, int *out) {
  return handle_errors(out, [&]) {
    return get_num_replicas(deref_opaque(parallel_tensor_shape));
  }
}

flexflow_error_t
    flexflow_is_valid(flexflow_parallel_tensor_shape_t parallel_tensor_shape,
                      bool *out) {
  return handle_errors(out, [&]) {
    return is_valid(deref_opaque(parallel_tensor_shape));
  }
}

flexflow_error_t flexflow_get_tensor_shape_unsafe(
    flexflow_parallel_tensor_shape_t parallel_tensor_shape,
    flexflow_tensor_shape_t *out) {
  return handle_errors(out, [&]) {
    return get_tensor_shape_unsafe(deref_opaque(parallel_tensor_shape));
  }
}

flexflow_error_t
    flexflow_get_tensor_shape_unsafe(flexflow_parallel_tesor_shape_t *input,
                                     int num_input,
                                     flexflow_tensor_shape_list_t *out) {
  return handle_errors(out, [&]) {
    return get_tensor_shape_unsafe(c_deref_opaque_list(input, num_input));
  }
}

flexflow_opattrs_error_t
    flexflow_get_datatype_size(flexflow_datatype_t datatype, int *out) {
  return handle_errors(out, [&]) {
    return size_of(to_internal(datatype));
  }
}

flexflow_opattrs_error_t
    flexflow_operator_attrs_get_op_type(flexflow_operator_attrs_t op_attrs,
                                        flexflow_op_type_t *out) {
  return handle_errors(out, [&]) {
    return deref_opaque(op_attrs).op_type;
  }
}

ParamSync to_internal(flexflow_param_sync_t e) {
  return to_internal_impl(e);
}

flexflow_param_sync_t to_external(ParamSync i) {
  return to_external_impl(i);
}

DataType to_internal(flexflow_datatype_t e) {
  return to_internal_impl(e);
}
flexflow_datatype_t to_external(DataType i) {
  return to_external_impl(i);
}

optional<Activation> to_internal(flexflow_activation_t e) {
  return to_internal_impl(e);
}

flexflow_activation_t to_external(optional<Activation> i) {
  return to_external_impl(i);
}

PoolOp to_internal(flexflow_pool_op_t e) {
  return to_internal_impl(e);
}
flexflow_pool_op_t to_external(PoolOp i) {
  return to_external_impl(i);
}

AggregateOp to_internal(flexflow_aggregate_op_t e) {
  return to_internal_impl(e);
}
flexflow_aggregate_op_t to_external(AggregateOp i) {
  return to_external_impl(i);
}

OperatorType to_internal(flexflow_op_type_t e) {
  return to_internal_impl(e);
}
flexflow_op_type_t to_external(OperatorType i) {
  return to_external_impl(i);
}
