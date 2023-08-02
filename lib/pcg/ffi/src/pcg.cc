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

flexflow_error_t flexflow_pcg_error_unwrap(flexflow_error_t err,
                                           flexflow_pcg_error_t *out) {
  return flexflow_error_unwrap(err, FLEXFLOW_ERROR_SOURCE_PCG, out);
}

flexflow_error_t flexflow_pcg_error_is_ok(flexflow_pcg_error_t, bool *out) {
  *out = false;
  return status_ok();
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
  try {
    *out = new_opaque<flexflow_computation_graph_t>();
    return status_ok();
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }
}

flexflow_error_t flexflow_computation_graph_destroy(flexflow_computation_graph_t opaque) {
  try {
    return status_ok();
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  } 
}

template <typename T>
std::string get_v1_string(T const &t) {
  return json{to_v1(t)}.dump();
}

template <typename T>
T from_v1_string(char const *s) {
  auto v1 = json::parse(s).template get<decltype(to_v1(std::declval<opaque_to_underlying_t<T>>()))>();
  return new_opaque<T>(from_v1(v1));
}

flexflow_error_t flexflow_computation_graph_serialize_to_buffer(flexflow_computation_graph_t opaque, char **out) {
  try {
    ComputationGraph const *cg = unwrap_opaque(opaque);

    std::string json_str = get_v1_string(*cg);

    *out = new char[json_str.size()+1];
    strncpy(*out, json_str.c_str(), json_str.size()+1);
    assert((*out)[json_str.size()] == '\x00');
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_computation_graph_deserialize_from_buffer(char *buf, flexflow_computation_graph_t *out) {
  try {
    *out = from_v1_string<flexflow_computation_graph_t>(buf);
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_computation_graph_serialize_to_file(flexflow_computation_graph_t cg, FILE *f) {
  try {
    if (f == nullptr) {
      return make_pcg_error(FLEXFLOW_PCG_STATUS_INVALID_FILE_PTR);
    }

    std::string json_str = get_v1_string(*unwrap_opaque(cg));

    size_t num_bytes_written = fwrite(json_str.c_str(), sizeof(char), json_str.size(), f);
    if (num_bytes_written < json_str.size()) {
      return make_pcg_error(FLEXFLOW_PCG_STATUS_FILE_WRITE_FAILED);
    }
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_computation_graph_deserialize_from_file(FILE *f, flexflow_computation_graph_t *out) {
  size_t constexpr BUF_SIZE = 256;

  try {
    char buf[BUF_SIZE];
    std::ostringstream oss; 

    while (true) {
      size_t num_bytes_read = fread(buf, sizeof(char), BUF_SIZE, f);
      oss.write(buf, num_bytes_read);
      if (num_bytes_read < BUF_SIZE) {
        if (feof(f)) {
          break;
        } else {
          return make_pcg_error(FLEXFLOW_PCG_STATUS_FILE_READ_FAILED);
        }
      } 
    }

    *out = from_v1_string<flexflow_computation_graph_t>(oss.str().c_str());
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_create(flexflow_computation_graph_t cg_handle,
                                        int num_dims,
                                        int *dims,
                                        flexflow_datatype_t datatype,
                                        bool create_grad,
                                        flexflow_tensor_t *out) {
  try {
    ComputationGraph *cg = unwrap_opaque(cg_handle);
    TensorDims ordered_dims{dims, dims + num_dims};
    TensorShape shape = { ordered_dims, to_internal(datatype) };
    Tensor result = create_tensor(*cg, shape, create_grad);
    *out = new_opaque<flexflow_tensor_t>(result);
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_get_create_grad(flexflow_tensor_t opaque, bool *out) {
  try {
    Tensor const *t = unwrap_opaque(opaque);
    *out = t->create_gradients;
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_get_initializer(flexflow_tensor_t opaque, flexflow_initializer_t *out) {
  try {
    Tensor const *t = unwrap_opaque(opaque);
    *out = new_opaque<flexflow_initializer_t>(t->initializer);
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_get_sync_type(flexflow_tensor_t opaque, flexflow_param_sync_t *out) {
  try {
    Tensor const *t = unwrap_opaque(opaque);
    *out = to_external(t->sync_type);
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_get_datatype(flexflow_tensor_t opaque, flexflow_datatype_t *out) {
  try {
    Tensor const *t = unwrap_opaque(opaque);
    *out = to_external(t->data_type);
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_get_num_dims(flexflow_tensor_t opaque, int *out) {
  try {
    Tensor const *t = unwrap_opaque(opaque);
    *out = t->num_dims();
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_get_dims(flexflow_tensor_t opaque, int *out) {
  try {
    Tensor const *t = unwrap_opaque(opaque);
    for (int i = 0; i < t->num_dims(); i++) {
      out[i] = t->dims.at(ff_dim_t(i));
    }
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_tensor_destroy(flexflow_tensor_t opaque) {
  try {
    delete_opaque(opaque);
  } catch (flexflow_utils_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
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
                                                          flexflow_computation_graph_t opaque_cg,
                                                          flexflow_tensor_t opaque_lhs,
                                                          flexflow_tensor_t opaque_rhs,
                                                          flexflow_tensor_t *out,
                                                          char *name) {
  try {
    ComputationGraph *cg = unwrap_opaque(opaque_cg);
    Tensor const *lhs = unwrap_opaque(opaque_lhs);
    Tensor const *rhs = unwrap_opaque(opaque_rhs);
    Tensor output = f(*cg, *lhs, *rhs, maybe_string(name));
    *out = new_opaque<flexflow_tensor_t>(output);
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t add_op(UnaryFunc const &f,
                                                   flexflow_computation_graph_t opaque_cg,
                                                   flexflow_tensor_t opaque_input,
                                                   flexflow_tensor_t *out,
                                                   char *name) {
  try {
    ComputationGraph *cg = unwrap_opaque(opaque_cg);
    Tensor const *input = unwrap_opaque(opaque_input);
    Tensor output = f(*cg, *input, maybe_string(name));
    *out = new_opaque<flexflow_tensor_t>(output);
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t add_op(ScalarUnaryFunc const &f,
                                                   flexflow_computation_graph_t opaque_cg,
                                                   flexflow_tensor_t opaque_input,
                                                   float scalar,
                                                   flexflow_tensor_t *out,
                                                   char *name) {
  UnaryFunc ff = [&](ComputationGraph &cg,
                     Tensor const &input,
                     optional<std::string> const &name) -> Tensor {
    return f(cg, input, scalar, name);
  };
  return add_op(ff, opaque_cg, opaque_input, out, name);
                      
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
  try {
    Tensor output = conv2d(deref_opaque(cg),
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
                           maybe_string(name));
    *out = new_opaque<flexflow_tensor_t>(output);
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}

flexflow_error_t flexflow_computation_graph_add_op_dropout(flexflow_computation_graph_t opaque_cg,
                                                           flexflow_tensor_t opaque_input,
                                                           float rate,
                                                           flexflow_tensor_t *out,
                                                           unsigned long long seed,
                                                           char *name) {
  UnaryFunc f = [&](ComputationGraph &cg,
                    Tensor const &input,
                    optional<std::string> const &name) -> Tensor {
    return dropout(cg, input, rate, seed, name);
  };
  return add_op(f, opaque_cg, opaque_input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_embedding(flexflow_computation_graph_t opaque_cg,
                                                             flexflow_tensor_t opaque_input,
                                                             int num_entries,
                                                             int out_dim,
                                                             flexflow_aggregate_op_t aggr_op,
                                                             flexflow_tensor_t *out,
                                                             flexflow_datatype_t output_type,
                                                             flexflow_initializer_t initializer,
                                                             char *name) {
  UnaryFunc f = [&](ComputationGraph &cg,
                    Tensor const &input,
                    optional<std::string> const &name) -> Tensor {
    return embedding(cg, input, num_entries, out_dim, to_internal(aggr_op), to_internal(output_type), c_deref_opaque(initializer), name);
  };
  return add_op(f, opaque_cg, opaque_input, out, name);
}

flexflow_error_t flexflow_computation_graph_add_op_gather(flexflow_computation_graph_t cg,
                                                          flexflow_tensor_t input,
                                                          flexflow_tensor_t index,
                                                          int dim,
                                                          flexflow_tensor_list_t *out,
                                                          char *name) {
  try {
    std::vector<Tensor> outputs = gather(deref_opaque(cg), 
                  c_deref_opaque(input),
                  c_deref_opaque(index),
                  ff_dim_t(dim),
                  maybe_string(name));
    *out = new_opaque<flexflow_tensor_list_t>(outputs);
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  } 

  return status_ok();
}

flexflow_error_t flexflow_computation_graph_add_op_group_by(flexflow_computation_graph_t cg,
                                                            flexflow_tensor_t data,
                                                            flexflow_tensor_t assign,
                                                            int n,
                                                            float alpha,
                                                            flexflow_tensor_list_t *out,
                                                            char *name) {
  try {
    std::vector<Tensor> outputs = group_by(deref_opaque(cg), 
                  c_deref_opaque(data),
                  c_deref_opaque(assign),
                  n, 
                  alpha,
                  maybe_string(name));
    *out = new_opaque<flexflow_tensor_list_t>(outputs);
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  } 

  return status_ok();
}

/* flexflow_error_t flexflow_computation_graph_add_op_cache() */
