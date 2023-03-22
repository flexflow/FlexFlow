#ifndef _FLEXFLOW_COMBINE_ATTRS_H
#define _FLEXFLOW_COMBINE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct CombineAttrs : public UnaryOpAttrs {
  ParallelTensorShape output_shape(ParallelTensorShape const &) const override;
  bool is_valid(ParallelTensorShape const &input) const override;
  OperatorType op_type() const override;
public:
  int combine_legion_dim;
  int combine_degree;
};
bool operator==(CombineAttrs const &, CombineAttrs const &);
bool operator<(CombineAttrs const &, CombineAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::CombineAttrs, combine_legion_dim, combine_degree);

namespace std {
template <>
struct hash<::FlexFlow::CombineAttrs> {
  size_t operator()(::FlexFlow::CombineAttrs const &) const;
};
} 

#endif 
