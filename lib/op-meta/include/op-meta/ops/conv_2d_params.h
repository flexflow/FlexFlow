#ifndef _FLEXFLOW_CONV_2D_PARAMS_H
#define _FLEXFLOW_CONV_2D_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct Conv2DParams {
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  ActiMode activation;
  bool use_bias;

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelTensorShape &output_shape,
                  ParallelTensorShape &kernel_shape,
                  ParallelTensorShape &bias_shape) const;

  friend bool operator==(Conv2DParams const &lhs, Conv2DParams const &rhs);

private:
  void mark_replica_dims(ParallelTensorShape const &input,
                         ParallelTensorShape &output_shape,
                         ParallelTensorShape &kernel_shape,
                         ParallelTensorShape &bias_shape) const;
  int output_size(ParallelTensorShape const &input,
                  ParallelTensorShape &output_shape) const;
  int kernel_size(ParallelTensorShape const &input_shape,
                  ParallelTensorShape &kernel_shape) const;
  int bias_size(ParallelTensorShape const &input,
                ParallelTensorShape &bias_shape) const;
};

} 

namespace std {
template <>
struct hash<FlexFlow::Conv2DParams> {
  size_t operator()(FlexFlow::Conv2DParams const &) const;
};
}

#endif // _FLEXFLOW_CONV_2D_PARAMS_H
