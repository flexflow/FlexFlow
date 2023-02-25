#ifndef _FLEXFLOW_RESHAPE_PARAMS_H
#define _FLEXFLOW_RESHAPE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ReshapeParams : public UnaryOpParams {
public:
  bool is_valid(ParallelTensorShape const &) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> shape;
};

bool operator==(ReshapeParams const &, ReshapeParams const &);
bool operator<(ReshapeParams const &, ReshapeParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ReshapeParams, shape);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ReshapeParams> {
  size_t operator()(::FlexFlow::opmeta::ReshapeParams const &) const;
};
}

#endif // _FLEXFLOW_RESHAPE_PARAMS_H
