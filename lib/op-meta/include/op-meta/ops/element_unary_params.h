#ifndef _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct ElementUnaryParams : public UnaryOpParams {
public:
  using AsConstTuple = std::tuple<OperatorType, bool, float>;
  AsConstTuple as_tuple() const;

  OperatorType op_type() const override;
public:
  OperatorType op;
  bool inplace;
  float scalar = 0.0;
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
