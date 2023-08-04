#include "pcg.h"
#include "flexflow/pcg.h"
#include "flexflow/utils.h"
#include "internal/opaque.h"
#include "internal/error.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/file_format/v1/graphs.h"
#include "utils/exception.h"
#include "internal/op-attrs.h"

using namespace FlexFlow;

flexflow_error_t flexflow_pcg_error_wrap(flexflow_pcg_error_t e) {
  return flexflow_error_wrap(FLEXFLOW_ERROR_SOURCE_PCG, *unwrap_opaque(e));
}

flexflow_error_t make_pcg_error(flexflow_pcg_error_code_t);
flexflow_ffi_exception_t make_pcg_exception(flexflow_pcg_error_code_t);

flexflow_error_t flexflow_pcg_error_unwrap(flexflow_error_t err,
                                           flexflow_pcg_error_t *out) {
  return flexflow_error_unwrap(err, FLEXFLOW_ERROR_SOURCE_PCG, out);
}

flexflow_error_t flexflow_pcg_error_is_ok(flexflow_pcg_error_t, bool *out) {
  *out = false;
  return status_ok();
}

template <typename T>
std::vector<T> make_vector(T const *ptr, int num_values) {
  if (num_values < 0) {
    throw make_pcg_exception(FLEXFLOW_PCG_STATUS_NEGATIVE_ARRAY_LENGTH_FOUND);
  }

  return {ptr, ptr+num_values};
}

flexflow_error_t flexflow_pcg_error_get_string(flexflow_pcg_error_t err, char **m_out) {
  flexflow_pcg_error_code_t err_code;
  RAISE_FLEXFLOW(flexflow_pcg_error_get_error_code(err, &err_code));

  auto out = const_cast<char const **>(m_out);

  switch (err_code) {
    case FLEXFLOW_PCG_ERROR_UNKNOWN:
    {
      *out = "Unknown error";
      break;
    }
    default:
      return make_pcg_error(FLEXFLOW_PCG_STATUS_INVALID_ERROR_CODE);
  }

  return status_ok();
}

flexflow_error_t flexflow_pcg_error_get_error_code(flexflow_error_t err, flexflow_pcg_error_code_t *out) {
  flexflow_pcg_error_t opaque;
  RAISE_FLEXFLOW(flexflow_pcg_error_unwrap(err, &opaque));
  internal_flexflow_pcg_error_t const *unwrapped = unwrap_opaque(opaque);
  *out = unwrapped->err_code;

  return status_ok();
}

flexflow_error_t flexflow_pcg_error_destroy(flexflow_pcg_error_t) {
  return status_ok();
}

flexflow_error_t flexflow_computation_graph_create(flexflow_computation_graph_t *out) {
  return handle_errors(out, [&] { return ComputationGraph{}; });
}

flexflow_error_t flexflow_computation_graph_destroy(flexflow_computation_graph_t opaque) {
  return handle_errors([&] { });
}

template <typename T>
std::string get_v1_string(T const &t) {
  return json{to_v1(t)}.dump();
}

template <typename T>
T from_v1_string(char const *s) {
  return json::parse(s).template get<decltype(to_v1(std::declval<opaque_to_underlying_t<T>>()))>();
}

flexflow_error_t flexflow_computation_graph_serialize_to_buffer(flexflow_computation_graph_t opaque, char **out) {
  return handle_errors([&] {
    ComputationGraph const *cg = unwrap_opaque(opaque);

    std::string json_str = get_v1_string(*cg);

    *out = new char[json_str.size()+1];
    strncpy(*out, json_str.c_str(), json_str.size()+1);
    assert((*out)[json_str.size()] == '\x00');
  });
}

flexflow_error_t flexflow_computation_graph_deserialize_from_buffer(char *buf, flexflow_computation_graph_t *out) {
  return handle_errors(out, [&] { return from_v1_string<ComputationGraph>(buf); });
}

flexflow_error_t flexflow_computation_graph_serialize_to_file(flexflow_computation_graph_t cg, FILE *f) {
  return handle_errors([&] {
    if (f == nullptr) {
      throw make_pcg_exception(FLEXFLOW_PCG_STATUS_INVALID_FILE_PTR);
    }

    std::string json_str = get_v1_string(*unwrap_opaque(cg));

    size_t num_bytes_written = fwrite(json_str.c_str(), sizeof(char), json_str.size(), f);
    if (num_bytes_written < json_str.size()) {
      throw make_pcg_exception(FLEXFLOW_PCG_STATUS_FILE_WRITE_FAILED);
    }
  });
}

flexflow_error_t flexflow_computation_graph_deserialize_from_file(FILE *f, flexflow_computation_graph_t *out) {
  return handle_errors(out, [&] {
    size_t constexpr BUF_SIZE = 256;

    char buf[BUF_SIZE];
    std::ostringstream oss; 

    while (true) {
      size_t num_bytes_read = fread(buf, sizeof(char), BUF_SIZE, f);
      oss.write(buf, num_bytes_read);
      if (num_bytes_read < BUF_SIZE) {
        if (feof(f)) {
          break;
        } else {
          throw make_pcg_exception(FLEXFLOW_PCG_STATUS_FILE_READ_FAILED);
        }
      } 
    }

    return from_v1_string<ComputationGraph>(oss.str().c_str());
  });
}

flexflow_error_t flexflow_tensor_create(flexflow_computation_graph_t cg,
                                        int num_dims,
                                        int *dims,
                                        flexflow_datatype_t datatype,
                                        bool create_grad,
                                        flexflow_tensor_t *out) {
  return handle_errors(out, [&] { 
    TensorDims ordered_dims{make_vector(dims, num_dims)};
    TensorShape shape = { ordered_dims, to_internal(datatype) };
    return create_tensor(deref_opaque(cg), shape, create_grad);
  });
}

flexflow_error_t flexflow_tensor_get_create_grad(flexflow_tensor_t opaque, bool *out) {
  return handle_errors(out, [&] { return c_deref_opaque(opaque).create_gradients; });
}

flexflow_error_t flexflow_tensor_get_initializer(flexflow_tensor_t opaque, flexflow_initializer_t *out) {
  return handle_errors(out, [&] { return c_deref_opaque(opaque).initializer; });
}

flexflow_error_t flexflow_tensor_get_sync_type(flexflow_tensor_t opaque, flexflow_param_sync_t *out) {
  return handle_errors(out, [&] { return c_deref_opaque(opaque).sync_type; });
}

flexflow_error_t flexflow_tensor_get_datatype(flexflow_tensor_t opaque, flexflow_datatype_t *out) {
  return handle_errors(out, [&] { return c_deref_opaque(opaque).data_type; });
}

flexflow_error_t flexflow_tensor_get_num_dims(flexflow_tensor_t opaque, int *out) {
  return handle_errors(out, [&] { return c_deref_opaque(opaque).num_dims(); });
}

flexflow_error_t flexflow_tensor_get_dims(flexflow_tensor_t opaque, int *out) {
  return handle_errors(out, [&] { return c_deref_opaque(opaque).dims; });
}

flexflow_error_t flexflow_tensor_destroy(flexflow_tensor_t opaque) {
  return handle_errors([&] { delete_opaque(opaque); });
}

optional<std::string> maybe_string(char *);
using BinaryFunc = std::function<Tensor(ComputationGraph &, Tensor const &, Tensor const &, optional<std::string> const &)>;
using UnaryFunc = std::function<Tensor(ComputationGraph &, Tensor const &, optional<std::string>)>;
using ScalarUnaryFunc = std::function<Tensor(ComputationGraph &, Tensor const &, float, optional<std::string>)>;

#define BINARY_OP(func) \
  [](ComputationGraph &cg, Tensor const &lhs, Tensor const &rhs, optional<std::string> const &name) -> Tensor { return func(cg, lhs, rhs, name); }

#define UNARY_OP(func) \
  [](ComputationGraph &cg, Tensor const &t, optional<std::string> const &name) -> Tensor { return func(cg, t, name); }

#define SCALAR_UNARY_OP(func) \
  [](ComputationGraph &cg, Tensor const &t, float scalar, optional<std::string> const &name) -> Tensor { return func(cg, t, scalar, name); }


flexflow_error_t add_op(BinaryFunc const &f,
                        flexflow_computation_graph_t cg,
                        flexflow_tensor_t lhs,
                        flexflow_tensor_t rhs,
                        flexflow_tensor_t *out,
                        char *name) {
  return handle_errors(out, [&] { return f(deref_opaque(cg),
                                           c_deref_opaque(lhs),
                                           c_deref_opaque(rhs),
                                           maybe_string(name)); });
}

flexflow_error_t add_op(UnaryFunc const &f,
                                                   flexflow_computation_graph_t cg,
                                                   flexflow_tensor_t input,
                                                   flexflow_tensor_t *out,
                                                   char *name) {
  return handle_errors(out, [&] { return f(deref_opaque(cg),
                                           c_deref_opaque(input),
                                           maybe_string(name)); });
}

flexflow_error_t add_op(ScalarUnaryFunc const &f,
                                                   flexflow_computation_graph_t opaque_cg,
                                                   flexflow_tensor_t opaque_input,
                                                   float scalar,
                                                   flexflow_tensor_t *out,
                                                   char *raw_name) {
  UnaryFunc ff = [&](ComputationGraph &cg,
                     Tensor const &input,
                     optional<std::string> const &name) -> Tensor {
    return f(cg, input, scalar, name);
  };
  return add_op(ff, opaque_cg, opaque_input, out, raw_name);
                      
}

flexflow_error_t flexflow_computation_graph_add_op_exp(flexflow_computation_graph_t opaque_cg,
                                                       flexflow_tensor_t opaque_input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(exp), opaque_cg, opaque_input, out, name);
}


flexflow_error_t flexflow_computation_graph_add_op_add(flexflow_computation_graph_t opaque_cg,
                                                       flexflow_tensor_t opaque_lhs,
                                                       flexflow_tensor_t opaque_rhs,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(BINARY_OP(add), opaque_cg, opaque_lhs, opaque_rhs, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_subtract(flexflow_computation_graph_t cg,
                                                            flexflow_tensor_t lhs,
                                                            flexflow_tensor_t rhs,
                                                            flexflow_tensor_t *out,
                                                            char *name) {
  return add_op(BINARY_OP(subtract), cg, lhs, rhs, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_multiply(flexflow_computation_graph_t cg,
                                                            flexflow_tensor_t lhs,
                                                            flexflow_tensor_t rhs,
                                                            flexflow_tensor_t *out,
                                                            char *name) {
  return add_op(BINARY_OP(multiply), cg, lhs, rhs, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_divide(flexflow_computation_graph_t cg,
                                                          flexflow_tensor_t lhs, 
                                                          flexflow_tensor_t rhs,
                                                          flexflow_tensor_t *out,
                                                          char *name) {
  return add_op(BINARY_OP(divide), cg, lhs, rhs, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_max(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t lhs,
                                                       flexflow_tensor_t rhs,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(BINARY_OP(max), cg, lhs, rhs, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_min(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t lhs,
                                                       flexflow_tensor_t rhs,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(BINARY_OP(min), cg, lhs, rhs, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_rsqrt(flexflow_computation_graph_t cg,
                                                         flexflow_tensor_t input,
                                                         flexflow_tensor_t *out,
                                                         char *name) {
  return add_op(UNARY_OP(rsqrt), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_scalar_multiply(flexflow_computation_graph_t cg,
                                                                   flexflow_tensor_t input,
                                                                   float scalar,
                                                                   flexflow_tensor_t *out,
                                                                   char *name) {
  return add_op(SCALAR_UNARY_OP(scalar_multiply), cg, input, scalar, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_scalar_add(flexflow_computation_graph_t cg,
                                                              flexflow_tensor_t input,
                                                              float scalar,
                                                              flexflow_tensor_t *out,
                                                              char *name) {
  return add_op(SCALAR_UNARY_OP(scalar_add), cg, input, scalar, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_scalar_sub(flexflow_computation_graph_t cg,
                                                              flexflow_tensor_t input,
                                                              float scalar,
                                                              flexflow_tensor_t *out,
                                                              char *name) {
  return add_op(SCALAR_UNARY_OP(scalar_sub), cg, input, scalar, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_scalar_truediv(flexflow_computation_graph_t cg,
                                                                  flexflow_tensor_t input,
                                                                  float scalar,
                                                                  flexflow_tensor_t *out,
                                                                  char *name) {
  return add_op(SCALAR_UNARY_OP(scalar_truediv), cg, input, scalar, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_sin(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(sin), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_cos(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(cos), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_relu(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(relu), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_identity(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(identity), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_gelu(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(gelu), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_sigmoid(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(sigmoid), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_tanh(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(tanh), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_elu(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       char *name) {
  return add_op(UNARY_OP(tanh), cg, input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_conv2d(
    flexflow_computation_graph_t cg,
    flexflow_tensor_t input,
    flexflow_tensor_t *out,
    int outChannels,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    flexflow_activation_t activation,
    int groups,
    bool use_bias,
    flexflow_initializer_t kernel_initializer,
    flexflow_initializer_t bias_initializer,
    flexflow_regularizer_attrs_t kernel_regularizer,
    char *name) {
  return handle_errors(out, [&] { return conv2d(deref_opaque(cg),
                           c_deref_opaque(input),
                           outChannels,
                           kernelH,
                           kernelW,
                           strideH,
                           strideW,
                           paddingH,
                           paddingW,
                           to_internal(activation),
                           groups,
                           use_bias,
                           c_deref_opaque(kernel_initializer),
                           c_deref_opaque(bias_initializer),
                           c_deref_opaque(kernel_regularizer),
                           maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_dropout(flexflow_computation_graph_t cg,
                                                           flexflow_tensor_t input,
                                                           flexflow_tensor_t *out,
                                                           float rate,
                                                           unsigned long long seed,
                                                           char *name) {
  return handle_errors(out, 
    [&] { return dropout(deref_opaque(cg),
                              c_deref_opaque(input),
                              rate,
                              seed,
                              maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_embedding(flexflow_computation_graph_t cg,
                                                             flexflow_tensor_t input,
                                                             flexflow_tensor_t *out,
                                                             int num_entries,
                                                             int out_dim,
                                                             flexflow_aggregate_op_t aggr_op,
                                                             flexflow_datatype_t output_type,
                                                             flexflow_initializer_t initializer,
                                                             char *name) {
  return handle_errors(out, 
    [&] { return embedding(deref_opaque(cg),
                                c_deref_opaque(input),
                                num_entries,
                                out_dim,
                                to_internal(aggr_op),
                                to_internal(output_type),
                                c_deref_opaque(initializer),
                                maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_gather(flexflow_computation_graph_t cg,
                                                          flexflow_tensor_t input,
                                                          flexflow_tensor_t index,
                                                          int dim,
                                                          flexflow_tensor_list_t *out,
                                                          char *name) {
  return handle_errors(out,
    [&] { return gather(deref_opaque(cg), 
                  c_deref_opaque(input),
                  c_deref_opaque(index),
                  ff_dim_t(dim),
                  maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_group_by(flexflow_computation_graph_t cg,
                                                            flexflow_tensor_t data,
                                                            flexflow_tensor_t assign,
                                                            int n,
                                                            float alpha,
                                                            flexflow_tensor_list_t *out,
                                                            char *name) {
  return handle_errors(out,
    [&] { return group_by(deref_opaque(cg), 
                  c_deref_opaque(data),
                  c_deref_opaque(assign),
                  n, 
                  alpha,
                  maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_cache(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    int num_batches,
    float (*score_func)(float *, void *, void *, int),
    flexflow_tensor_t assign,
    flexflow_tensor_t *outs,
    int n,
    float alpha,
    char *name) {
  NOT_IMPLEMENTED(); // TODO @lockshaw
}

flexflow_error_t flexflow_computation_graph_add_op_aggregate(flexflow_computation_graph_t cg, 
                                                             flexflow_tensor_t gate_preds,
                                                             flexflow_tensor_t gate_assign,
                                                             flexflow_tensor_t true_gate_assign,
                                                             flexflow_tensor_t full_gate_gradients,
                                                             flexflow_tensor_t *exp_preds,
                                                             int n,
                                                             float lambda_bal,
                                                             flexflow_tensor_t *out,
                                                             char *name) {
  return handle_errors(out,
    [&] { return aggregate(deref_opaque(cg),
                              c_deref_opaque(gate_preds),
                              c_deref_opaque(gate_assign),
                              c_deref_opaque(true_gate_assign),
                              c_deref_opaque(full_gate_gradients),
                              c_deref_opaque_list(exp_preds, n), 
                              lambda_bal,
                              maybe_string(name)); });
}

flexflow_error_t aggregate_spec(flexflow_computation_graph_t cg,
                      flexflow_tensor_t *inputs,
                      flexflow_tensor_t *out,
                      int n,
                      float lambda_bal,
                      char *name) {
  return handle_errors(out,
    [&] { return aggregate_spec(deref_opaque(cg),
                                   c_deref_opaque_list(inputs, n),
                                   lambda_bal,
                                   maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_pool2d(flexflow_computation_graph_t cg,
                                                          flexflow_tensor_t input,
                                                          flexflow_tensor_t *out,
                                                          int kernelH,
                                                          int kernelW,
                                                          int strideH,
                                                          int strideW,
                                                          int paddingH,
                                                          int paddingW,
                                                          flexflow_pool_op_t pool_op,
                                                          flexflow_activation_t activation,
                                                          char *name) {
  return handle_errors(out,
    [&] { return pool2d(deref_opaque(cg),
                             c_deref_opaque(input),
                             kernelH,
                             kernelW,
                             strideH,
                             strideW,
                             paddingH,
                             paddingW,
                             to_internal(pool_op),
                             to_internal(activation),
                             maybe_string(name)); });
}

flexflow_error_t flexcflow_computation_graph_add_op_layer_norm(flexflow_computation_graph_t cg,
                                                               flexflow_tensor_t input,
                                                               flexflow_tensor_t *out,
                                                               int *axes,
                                                               int num_axes,
                                                               bool elementwise_affine,
                                                               float eps,
                                                               char *name) {
  return handle_errors(out,
    [&] { return layer_norm(deref_opaque(cg),
                                 c_deref_opaque(input),
                                 make_vector(axes, num_axes),
                                 elementwise_affine,
                                 eps,
                                 maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_batch_norm(flexflow_computation_graph_t cg,
                                                              flexflow_tensor_t input,
                                                              flexflow_tensor_t *out,
                                                              bool relu,
                                                              char *name) {

  return handle_errors(out,
    [&] { return batch_norm(deref_opaque(cg),
                                 c_deref_opaque(input),
                                 relu,
                                 maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_batch_matmul(flexflow_computation_graph_t cg,
                                                                flexflow_tensor_t lhs,
                                                                flexflow_tensor_t rhs,
                                                                flexflow_tensor_t *out,
                                                                int a_seq_length_dim,
                                                                int b_seq_length_dim,
                                                                char *name) {
  return handle_errors(out,
    [&] { return batch_matmul(deref_opaque(cg),
                                   c_deref_opaque(lhs),
                                   c_deref_opaque(rhs),
                                   a_seq_length_dim,
                                   b_seq_length_dim,
                                   maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_dense(flexflow_computation_graph_t cg,
                                                         flexflow_tensor_t input,
                                                         flexflow_tensor_t *out,
                                                         int out_dim,
                                                         flexflow_activation_t activation,
                                                         bool use_bias,
                                                         flexflow_datatype_t compute_type,
                                                         flexflow_initializer_t kernel_initializer,
                                                         flexflow_initializer_t bias_initializer,
                                                         char *name) {
  return handle_errors(out,
    [&] { return dense(deref_opaque(cg),
                            c_deref_opaque(input),
                            out_dim,
                            to_internal(activation),
                            use_bias,
                            to_internal(compute_type),
                            deref_opaque(kernel_initializer),
                            deref_opaque(bias_initializer),
                            maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_cast(flexflow_computation_graph_t cg,
                                                        flexflow_tensor_t input,
                                                        flexflow_tensor_t *out,
                                                        flexflow_datatype_t out_type,
                                                        char *name) {
  return handle_errors(out, [&] { return cast(deref_opaque(cg),
                           c_deref_opaque(input),
                           to_internal(out_type),
                           maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_concat(flexflow_computation_graph_t cg,
                                                          flexflow_tensor_t *inputs,
                                                          flexflow_tensor_t *out,
                                                          int num_inputs,
                                                          int axis,
                                                          char *name) {
  return handle_errors(out, [&] { return concat(deref_opaque(cg),
                           c_deref_opaque_list(inputs, num_inputs),
                           axis,
                           maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_mean(flexflow_computation_graph_t cg,
                                                        flexflow_tensor_t input,
                                                        flexflow_tensor_t *out,
                                                        int *dims,
                                                        int num_dims, 
                                                        bool keepdims,
                                                        char *name) {
  return handle_errors(out, [&] { return mean(deref_opaque(cg),
                           c_deref_opaque(input),
                           make_vector(dims, num_dims),
                           keepdims,
                           maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_moe(flexflow_computation_graph_t cg,
                                                       flexflow_tensor_t input,
                                                       flexflow_tensor_t *out,
                                                       int num_exp,
                                                       int num_select,
                                                       int expert_hidden_size,
                                                       float alpha,
                                                       float lambda,
                                                       char *name) {
  return handle_errors(out, [&] { return 
      moe(deref_opaque(cg),
          c_deref_opaque(input),
          num_exp,
          num_select,
          expert_hidden_size,
          alpha,
          lambda,
          maybe_string(name)
      ); }
    );
}

flexflow_error_t flexflow_computation_graph_add_op_split(flexflow_computation_graph_t cg,
                                                         flexflow_tensor_t input,  
                                                         flexflow_tensor_list_t *out,
                                                         int *splits,
                                                         int num_splits,
                                                         int axis,
                                                         char *name) {
  return handle_errors(out, [&] { return 
      split(deref_opaque(cg),
            c_deref_opaque(input),
            make_vector(splits, num_splits),
            axis,
            maybe_string(name)
      ); }
    );
}

flexflow_error_t flexflow_computation_graph_add_op_flat(flexflow_computation_graph_t cg,
                                                        flexflow_tensor_t input,
                                                        flexflow_tensor_t *out,
                                                        char *name) {
  return handle_errors(out, [&] { return 
      flat(deref_opaque(cg),
           c_deref_opaque(input),
           maybe_string(name)
      );
      });
}

flexflow_error_t flexflow_computation_graph_add_op_softmax(flexflow_computation_graph_t cg,
                                                           flexflow_tensor_t input,
                                                           flexflow_tensor_t *out,
                                                           int dim,
                                                           char *name) {
  return handle_errors(out, [&] { return softmax(deref_opaque(cg),
                              c_deref_opaque(input),
                              dim,
                              maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_transpose(flexflow_computation_graph_t cg,
                                                             flexflow_tensor_t input,
                                                             flexflow_tensor_t *out,
                                                             int *permutation,
                                                             int num_permutation_values,
                                                             char *name) {
  return handle_errors(out, [&] { return transpose(deref_opaque(cg),
                                c_deref_opaque(input),
                                make_vector(permutation, num_permutation_values),
                                maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_reduce_sum(flexflow_computation_graph_t cg,
                                                              flexflow_tensor_t input,
                                                              flexflow_tensor_t *out,
                                                              int *axes,
                                                              int num_axes,
                                                              bool keepdims,
                                                              char *name) {
  return handle_errors(out, [&] { return reduce_sum(deref_opaque(cg),
                                 c_deref_opaque(input),
                                 make_vector(axes, num_axes),
                                 keepdims,
                                 maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_reshape(flexflow_computation_graph_t cg,
                                                           flexflow_tensor_t input,
                                                           flexflow_tensor_t *out,
                                                           int *shape,
                                                           int num_shape_entries,
                                                           char *name) {
  return handle_errors(out, [&] { return reshape(deref_opaque(cg),
                              c_deref_opaque(input),
                              make_vector(shape, num_shape_entries),
                              maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_reverse(flexflow_computation_graph_t cg,
                                                           flexflow_tensor_t input,
                                                           flexflow_tensor_t *out,
                                                           int axis,
                                                           char *name) {
  return handle_errors(out, [&] { return reverse(deref_opaque(cg),
                              c_deref_opaque(input),
                              axis,
                              maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_topk(flexflow_computation_graph_t cg,
                                                        flexflow_tensor_t input,
                                                        flexflow_tensor_list_t *out,
                                                        int k,
                                                        bool sorted,
                                                        char *name) {
  return handle_errors(out, [&] { return top_k(deref_opaque(cg),
                           c_deref_opaque(input),
                           k,
                           sorted,
                           maybe_string(name)); });
}

flexflow_error_t flexflow_computation_graph_add_op_multihead_attention(flexflow_computation_graph_t cg,
                                                                       flexflow_tensor_t query,
                                                                       flexflow_tensor_t key,
                                                                       flexflow_tensor_t value,
                                                                       flexflow_tensor_t *out,
                                                                       int embed_dim,
                                                                       int num_heads,
                                                                       int kdim,
                                                                       int vdim,
                                                                       float dropout,
                                                                       bool bias,
                                                                       bool add_bias_kv,
                                                                       bool add_zero_attn,
                                                                       flexflow_initializer_t initializer,
                                                                       char *name) {
  return handle_errors(out, [&] { return multihead_attention(deref_opaque(cg),
                                                             c_deref_opaque(query),
                                                             c_deref_opaque(key),
                                                             c_deref_opaque(value),
                                                             embed_dim,
                                                             num_heads,
                                                             kdim,
                                                             vdim,
                                                             dropout,
                                                             bias,
                                                             add_bias_kv,
                                                             add_zero_attn,
                                                             c_deref_opaque(initializer),
                                                             maybe_string(name)); });
}


/* flexflow_error_t flexflow_computation_graph_add_op_cache() */
