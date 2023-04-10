#include "model_spec.h"
#include "op-attrs/ffconst.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/expected.h"

namespace FlexFlow {

or_error_msg<Tensor> ModelSpec::as_type(Tensor const &x, DataType data_type, std::string const &name) {
  if (x->data_type < data_type) {
    return this->cast(x, data_type, name);
  } else if (x->data_type > data_type) {
    return error_msg("Could not convert provided tensor data type {} to desired data type {}",
                     x->data_type, data_type);
  }
  return x;
}

Tensor ModelSpec::element_unary(variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &attrs, Tensor const &x, std::string const &name) {
  Tensor input = this->as_type(x, DT_FLOAT, name + "input_pre_cast");
  Layer layer = this->layer_mgr.create(attrs, DT_FLOAT, name);  
  TensorShape output_shape = get_output_shape(attrs, input->get_shape());
  Tensor output = this->tensor_mgr.create(output_shape, CreateGrad::YES);

  this->add_layer(layer, {input}, {}, {output});

  return output;
}

Tensor ModelSpec::element_unary(OperatorType op_type, Tensor const &input, std::string const &name) {
  ElementUnaryAttrs attrs = { op_type };
  return this->element_unary(attrs, input, name);
}

Tensor ModelSpec::element_scalar_unary(OperatorType op_type, Tensor const &input, float scalar, std::string const &name) {
  ElementScalarUnaryAttrs attrs = { op_type, scalar };
  return this->element_unary(attrs, input, name);
}

Tensor ModelSpec::element_binary(OperatorType op_type, Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  TensorShape compute_shape = this->get_broadcast_target_shape({lhs->get_shape(), rhs->get_shape()});

  bool did_broadcast_lhs, did_broadcast_rhs;
  optional<Tensor> lhs_input, rhs_input;

  std::tie(did_broadcast_lhs, lhs_input) = this->broadcast(lhs, compute_shape);
  std::tie(did_broadcast_rhs, rhs_input) = this->broadcast(rhs, compute_shape);

  ElementBinaryAttrs attrs = { op_type, did_broadcast_lhs, did_broadcast_rhs };

  DataType compute_type = std::max(lhs->data_type, rhs->data_type);
  lhs_input = this->as_type(lhs_input.value(), compute_type, name + "inputl_pre_cast");
  rhs_input = this->as_type(rhs_input.value(), compute_type, name + "inputr_pre_cast");

  Layer layer = this->layer_mgr.create(attrs, compute_type, name);
  TensorShape output_shape = get_output_shape(attrs, lhs_input.value()->get_shape());
  Tensor output = this->tensor_mgr.create(output_shape, CreateGrad::YES);

  this->add_layer(layer, {lhs_input.value(), rhs_input.value()}, {}, {output});

  return output;
}

Tensor ModelSpec::exp(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_EXP, input, name);
}

Tensor ModelSpec::add(Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  return this->element_binary(OP_EW_ADD, lhs, rhs, name);
}

Tensor ModelSpec::subtract(Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  return this->element_binary(OP_EW_SUB, lhs, rhs, name);
}

Tensor ModelSpec::multiply(Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  return this->element_binary(OP_EW_MUL, lhs, rhs, name);
}

Tensor ModelSpec::divide(Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  return this->element_binary(OP_EW_DIV, lhs, rhs, name);
}

Tensor ModelSpec::max(Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  return this->element_binary(OP_EW_MAX, lhs, rhs, name);
}

Tensor ModelSpec::min(Tensor const &lhs, Tensor const &rhs, std::string const &name) {
  return this->element_binary(OP_EW_MIN, lhs, rhs, name);
}

Tensor ModelSpec::rsqrt(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_RSQRT, input, name);
}

Tensor ModelSpec::pow(Tensor const &input, float exponent, std::string const &name) {
  return this->element_scalar_unary(OP_POW, input, exponent, name);
}

Tensor ModelSpec::scalar_multiply(Tensor const &input, float scalar, std::string const &name) {
  return this->element_scalar_unary(OP_SCALAR_MULTIPLY, input, scalar, name);
}

Tensor ModelSpec::scalar_add(Tensor const &input, float scalar, std::string const &name) {
  return this->element_scalar_unary(OP_SCALAR_ADD, input, scalar, name);
}

Tensor ModelSpec::scalar_sub(Tensor const &lhs, float rhs, std::string const &name) {
  return this->element_scalar_unary(OP_SCALAR_SUB, lhs, rhs, name);
}

Tensor ModelSpec::scalar_truediv(Tensor const &numerator, float denominator, std::string const &name) {
  return this->element_scalar_unary(OP_SCALAR_TRUE_DIV, numerator, denominator, name);
}

Tensor ModelSpec::sin(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_SIN, input, name);
}

Tensor ModelSpec::cos(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_COS, input, name);
}

Tensor ModelSpec::relu(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_RELU, input, name);
}

Tensor ModelSpec::identity(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_IDENTITY, input, name);
}

Tensor ModelSpec::gelu(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_GELU, input, name);
}

Tensor ModelSpec::sigmoid(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_SIGMOID, input, name);
}

Tensor ModelSpec::tanh(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_TANH, input, name);
}

Tensor ModelSpec::elu(Tensor const &input, std::string const &name) {
  return this->element_unary(OP_ELU, input, name);
}

Tensor ModelSpec::conv2d(Tensor const &x, 
                         int outChannels,
                         int kernelH,
                         int kernelW,
                         int strideH,
                         int strideW,
                         int paddingH,
                         int paddingW,
                         std::string const &name,
                         ActiMode activation,
                         int groups,
                         bool use_bias,
                         Initializer *kernel_initializer, 
                         Initializer *bias_initializer) {
  Conv2DAttrs attrs = {
    outChannels,
    kernelH,
    kernelW,
    strideH, 
    strideW,
    paddingH,
    paddingW,
    groups,
    activation,
    use_bias
  };

  Tensor input = this->as_type(x, DT_FLOAT, name + "input_pre_cast");
  Layer layer = this->layer_mgr.create(attrs, DT_FLOAT, name);  
  TensorShape output_shape = get_output_shape(attrs, input->get_shape());

  std::vector<std::pair<TensorShape, Initializer *>> weights;

  weights.push_back({get_kernel_shape(attrs, input), kernel_initializer});

  if (use_bias) {
    weights.push_back({
      get_bias_shape(attrs, input),
      bias_initializer
    });
  }

  return this->add_layer(layer, {input}, weights, output_shape);
}

Tensor ModelSpec::dropout(Tensor const &x,
                          float rate,
                          std::string const &name, 
                          unsigned long long seed) {
  DropoutAttrs attrs = { rate, seed };

  Layer layer = this->layer_mgr.create(attrs, DT_FLOAT, name);
  Tensor input = this->as_type(x, DT_FLOAT, name + "input_pre_cast");

  TensorShape output_shape = get_output_shape(attrs, input);

  return this->add_layer(layer, {input}, {}, output_shape);
}

Tensor ModelSpec::embedding(Tensor const &x,
                            int num_entries, 
                            int outDim,
                            AggrMode aggr,
                            std::string const &name,
                            DataType dtype,
                            Initializer *kernel_initializer) {
  EmbeddingAttrs attrs = { num_entries, outDim, aggr, dtype };

  Layer layer = this->layer_mgr.create(attrs, DT_FLOAT, name);
  Tensor input = this->as_type(x, DT_FLOAT, name + "input_pre_cast");

  TensorShape output_shape = get_output_shape(attrs, input);
  TensorShape weights_shape = get_weights_shape(attrs, input);
  
  return this->add_layer(layer, {input}, {{weights_shape, kernel_initializer}}, output_shape);
}

Tensor ModelSpec::gather(Tensor const &data, 
                         Tensor const &assign,
                         int dim,
                         std::string const &name) {
  GatherAttrs attrs = { legion_dim_t(dim) };

  Layer layer = this->layer_mgr.create(attrs, DT_FLOAT, name);
  if () {
    
  }
}

}
