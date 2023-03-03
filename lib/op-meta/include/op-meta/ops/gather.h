#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "op-meta/parallel_tensor_shape.h"
#include "visit_struct/visit_struct.hpp"
#include "binary_op.h"

namespace FlexFlow {

struct GatherAttrs : public BinaryOpAttrs {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
  bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const override;
public:
  int legion_dim;
};

bool operator==(GatherAttrs const &, GatherAttrs const &);
bool operator<(GatherAttrs const &, GatherAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::GatherAttrs, legion_dim);

namespace std {
template <>
struct hash<::FlexFlow::GatherAttrs> {
  size_t operator()(::FlexFlow::GatherAttrs const &) const;
};
} 

#endif 
