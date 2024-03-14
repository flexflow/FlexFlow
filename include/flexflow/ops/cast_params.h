#ifndef _FLEXFLOW_CAST_PARAMS_H
#define _FLEXFLOW_CAST_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct CastParams {
  DataType dtype;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(CastParams const &, CastParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CastParams> {
  size_t operator()(FlexFlow::CastParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_CAST_PARAMS_H
