#ifndef _FLEXFLOW_SOFTMAX_PARAMS_H
#define _FLEXFLOW_SOFTMAX_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct SoftmaxParams : public UnaryOpParams {
public:
  using AsConstTuple = std::tuple<int>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &) const;
  OperatorType op_type() const;
public:
  int dim;
};
bool operator==(SoftmaxParams const &, SoftmaxParams const &);
bool operator<(SoftmaxParams const &, SoftmaxParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SoftmaxParams> {
  size_t operator()(FlexFlow::SoftmaxParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SOFTMAX_PARAMS_H
