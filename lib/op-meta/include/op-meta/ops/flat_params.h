#ifndef _FLEXFLOW_FLAT_PARAMS_H
#define _FLEXFLOW_FLAT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct FlatParams {
  bool is_valid(ParallelTensorShape const &) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

  using AsConstTuple = std::tuple<>;
  AsConstTuple as_tuple() const;

private:
  int output_size(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;
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
