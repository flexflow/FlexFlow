#ifndef _FLEXFLOW_ELEMENT_BINARY_PARAMS_H
#define _FLEXFLOW_ELEMENT_BINARY_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ElementBinaryParams {
  LayerID layer_guid;
  OperatorType type;
  bool inplace_a;
  char name[MAX_OPNAME];

  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(ElementBinaryParams const &, ElementBinaryParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ElementBinaryParams> {
  size_t operator()(FlexFlow::ElementBinaryParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ELEMENT_BINARY_PARAMS_H
