#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct LinearAttrs : public UnaryOpAttrs {
public:
  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_kernel_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_bias_shape(ParallelTensorShape const &input_shape) const;

  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;
};

bool operator==(LinearAttrs const &, LinearAttrs const &);
bool operator<(LinearAttrs const &, LinearAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::LinearAttrs, out_channels, use_bias, data_type, activation);

namespace std {
template <>
struct hash<::FlexFlow::LinearAttrs> {
  size_t operator()(::FlexFlow::LinearAttrs const &) const;
};
} 

#endif 
