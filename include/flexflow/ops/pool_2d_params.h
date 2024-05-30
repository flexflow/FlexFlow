#ifndef _FLEXFLOW_POOL_2D_PARAMS_H
#define _FLEXFLOW_POOL_2D_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct Pool2DParams {
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
  char name[MAX_OPNAME];

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims) const;

private:
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const;
};

bool operator==(Pool2DParams const &, Pool2DParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::Pool2DParams> {
  size_t operator()(FlexFlow::Pool2DParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_POOL_2D_PARAMS_H
