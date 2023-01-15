#ifndef _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct ElementUnaryParams {
  OperatorType op_type;
  bool inplace;
  float scalar = 0.0;

  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<OperatorType, bool, float>;
  AsConstTuple as_tuple() const;
};

bool operator==(ElementUnaryParams const &, ElementUnaryParams const &);
bool operator<(ElementUnaryParams const &, ElementUnaryParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ElementUnaryParams> {
  size_t operator()(FlexFlow::ElementUnaryParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H
