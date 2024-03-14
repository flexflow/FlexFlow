#ifndef _FLEXFLOW_CONV_2D_PARAMS_H
#define _FLEXFLOW_CONV_2D_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct Conv2DParams {
  LayerID layer_guid;
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  ActiMode activation;
  bool use_bias;
  char name[MAX_OPNAME];

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM],
                  int *kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM],
                  int *bias_ndims) const;

  friend bool operator==(Conv2DParams const &lhs, Conv2DParams const &rhs);

private:
  void mark_replica_dims(ParallelTensorShape const &input,
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const;
  int kernel_size(ParallelTensorShape const &input_shape,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM]) const;
  int bias_size(ParallelTensorShape const &input,
                ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
};

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::Conv2DParams> {
  size_t operator()(FlexFlow::Conv2DParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_CONV_2D_PARAMS_H
