#ifndef _FLEXFLOW_ELEMENT_BINARY_PARAMS_H
#define _FLEXFLOW_ELEMENT_BINARY_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct ElementBinaryParams {
  OperatorType type;

  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;

  using AsConstTuple = std::tuple<OperatorType>;
  AsConstTuple as_tuple() const;
};

bool operator==(ElementBinaryParams const &, ElementBinaryParams const &);
bool operator<(ElementBinaryParams const &, ElementBinaryParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ElementBinaryParams> {
  size_t operator()(FlexFlow::ElementBinaryParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ELEMENT_BINARY_PARAMS_H
