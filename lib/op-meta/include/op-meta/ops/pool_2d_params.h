#ifndef _FLEXFLOW_POOL_2D_PARAMS_H
#define _FLEXFLOW_POOL_2D_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct Pool2DParams {
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

private:
  int output_size(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;
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
