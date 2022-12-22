#ifndef _FLEXFLOW_OPS_PARAMS_NOOP_PARAMS_H
#define _FLEXFLOW_OPS_PARAMS_NOOP_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"
#include "mpark/variant.hpp"

namespace mp = mpark;

namespace FlexFlow {

struct NoOpParams {
  OperatorType op_type;
  tl::optional<ParallelTensorShape> input_metadata = tl::nullopt;

  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const;
};

bool operator==(NoOpParams const &, NoOpParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::NoOpParams> {
  size_t operator()(FlexFlow::NoOpParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_OPS_PARAMS_NOOP_PARAMS_H
