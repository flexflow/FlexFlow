#ifndef _FLEXFLOW_RESHAPE_PARAMS_H
#define _FLEXFLOW_RESHAPE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct ReshapeParams : public UnaryOpParams {
public:
  using AsConstTuple = std::tuple<std::vector<int>>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> shape;
};

bool operator==(ReshapeParams const &, ReshapeParams const &);
bool operator<(ReshapeParams const &, ReshapeParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReshapeParams> {
  size_t operator()(FlexFlow::ReshapeParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_RESHAPE_PARAMS_H
