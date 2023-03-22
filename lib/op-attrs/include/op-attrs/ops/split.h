#ifndef _FLEXFLOW_SPLIT_ATTRS_H
#define _FLEXFLOW_SPLIT_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct SplitAttrs : public UnaryInputOpAttrs {
public:
  std::vector<ParallelTensorShape> output_shapes(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> splits;
  int legion_axis;
};

bool operator==(SplitAttrs const &, SplitAttrs const &);
bool operator<(SplitAttrs const &, SplitAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::SplitAttrs, splits, legion_axis);

namespace std {
template <>
struct hash<::FlexFlow::SplitAttrs> {
  size_t operator()(::FlexFlow::SplitAttrs const &) const;
};
} 

#endif 
