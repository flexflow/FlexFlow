#ifndef _FLEXFLOW_CONV_2D_PARAMS_H
#define _FLEXFLOW_CONV_2D_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct Conv2DParams : public UnaryOpParams {
public:

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_kernel_shape(ParallelTensorShape const &input_shape) const;
  ParallelTensorShape calculate_bias_shape(ParallelTensorShape const &input_shape) const;

  /**
   * @brief Check if the given input shape is valid for this configuration
   *
   * Likely deprecated (see https://github.com/flexflow/FlexFlow/pull/317)
   */
  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  ActiMode activation;
  bool use_bias;
};
bool operator==(Conv2DParams const &, Conv2DParams const &);
bool operator<(Conv2DParams const &, Conv2DParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::Conv2DParams, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, activation, use_bias);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::Conv2DParams> {
  size_t operator()(::FlexFlow::opmeta::Conv2DParams const &) const;
};
}

#endif // _FLEXFLOW_CONV_2D_PARAMS_H
