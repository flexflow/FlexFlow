#ifndef _FLEXFLOW_CONCAT_ATTRS_H
#define _FLEXFLOW_CONCAT_ATTRS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_attrs.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ConcatAttrs : public OpAttrsInterface {
public:
  bool is_valid(std::vector<ParallelTensorShape> const &input_shapes) const override;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &input_shapes) const override;
  OperatorType op_type() const override;
public:
  int axis;
};

bool operator==(ConcatAttrs const &, ConcatAttrs const &);
bool operator<(ConcatAttrs const &, ConcatAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ConcatAttrs, axis);

namespace std {
template <>
struct hash<::FlexFlow::ConcatAttrs> {
  size_t operator()(::FlexFlow::ConcatAttrs const &) const;
};
} 

#endif // _FLEXFLOW_CONCAT_ATTRS_H
