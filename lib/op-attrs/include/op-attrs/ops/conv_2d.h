#ifndef _FLEXFLOW_CONV_2D_ATTRS_H
#define _FLEXFLOW_CONV_2D_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Conv2DAttrs {
public:
  Conv2DAttrs() = delete;
  Conv2DAttrs(int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups, ActiMode activation, bool use_bias);

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_kernel_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_bias_shape(ParallelTensorShape const &input_shape) const;

  /**
   * @brief Check if the given input shape is valid for this configuration
   *
   * Likely deprecated (see https://github.com/flexflow/FlexFlow/pull/317)
   */
  /* bool is_valid(ParallelTensorShape const &input_shape) const override; */
  /* ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override; */
  /* OperatorType op_type() const override; */
public:
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  ActiMode activation;
  bool use_bias;
};
bool operator==(Conv2DAttrs const &, Conv2DAttrs const &);
bool operator<(Conv2DAttrs const &, Conv2DAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::Conv2DAttrs, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, activation, use_bias);

namespace std {
template <>
struct hash<::FlexFlow::Conv2DAttrs> {
  size_t operator()(::FlexFlow::Conv2DAttrs const &) const;
};
}

#endif 
