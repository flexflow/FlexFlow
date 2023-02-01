#ifndef _FLEXFLOW_POOL_2D_PARAMS_H
#define _FLEXFLOW_POOL_2D_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct Pool2DParams : public OpParamsInterface {
public:
  void solve_dims(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input) const;
  
  using AsConstTuple = std::tuple<int, int, int, int, int, int, PoolType, ActiMode>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &input) const override;
  OperatorType op_type() const override;
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

bool operator==(Pool2DParams const &, Pool2DParams const &);
bool operator<(Pool2DParams const &, Pool2DParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::Pool2DParams> {
  size_t operator()(FlexFlow::Pool2DParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_POOL_2D_PARAMS_H
