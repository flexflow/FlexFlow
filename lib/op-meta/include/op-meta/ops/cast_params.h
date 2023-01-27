#ifndef _FLEXFLOW_CAST_PARAMS_H
#define _FLEXFLOW_CAST_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct CastParams {
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<DataType>;
  AsConstTuple as_tuple() const;

public:
  DataType dtype;
};

bool operator==(CastParams const &, CastParams const &);
bool operator<(CastParams const &, CastParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CastParams> {
  size_t operator()(FlexFlow::CastParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_CAST_PARAMS_H
