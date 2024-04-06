#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SigmoidSiluMultiParams {
  LayerID layer_guid;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(SigmoidSiluMultiParams const &, SigmoidSiluMultiParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SigmoidSiluMultiParams> {
  size_t operator()(FlexFlow::SigmoidSiluMultiParams const &) const;
};
} // namespace std
