#ifndef _FLEXFLOW_FLAT_PARAMS_H
#define _FLEXFLOW_FLAT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct FlatParams : public UnaryOpParams {
  using AsConstTuple = std::tuple<>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
private:
  int output_size(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input) const;
};

bool operator==(FlatParams const &, FlatParams const &);
bool operator<(FlatParams const &, FlatParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::FlatParams> {
  size_t operator()(FlexFlow::FlatParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_FLAT_PARAMS_H
