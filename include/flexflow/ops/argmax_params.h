#ifndef _FLEXFLOW_ARGMAX_PARAMS_H
#define _FLEXFLOW_ARGMAX_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ArgMaxParams {
  OperatorType op_type;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ArgMaxParams const &, ArgMaxParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ArgMaxParams> {
  size_t operator()(FlexFlow::ArgMaxParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ARGMAX_PARAMS_H