#ifndef _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ElementUnaryParams {
  OperatorType op_type;
  bool inplace;
  float scalar = 0.0;
  LayerID layer_guid;
  char name[MAX_OPNAME];

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(ElementUnaryParams const &, ElementUnaryParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ElementUnaryParams> {
  size_t operator()(FlexFlow::ElementUnaryParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ELEMENTARY_UNARY_PARAMS_H
