#ifndef _FLEXFLOW_DROPOUT_PARAMS_H
#define _FLEXFLOW_DROPOUT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct DropoutParams : public UnaryOpParams {
  using AsConstTuple = std::tuple<float, unsigned long long>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  float rate;
  unsigned long long seed;
};

bool operator==(DropoutParams const &, DropoutParams const &);
bool operator<(DropoutParams const &, DropoutParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::DropoutParams> {
  size_t operator()(FlexFlow::DropoutParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_DROPOUT_PARAMS_H
