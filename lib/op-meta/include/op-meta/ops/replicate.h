#ifndef _FLEXFLOW_REPLICATE_ATTRS_H
#define _FLEXFLOW_REPLICATE_ATTRS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ReplicateAttrs : public UnaryOpAttrs {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int replicate_legion_dim;
  int replicate_degree;
};

bool operator==(ReplicateAttrs const &, ReplicateAttrs const &);
bool operator<(ReplicateAttrs const &, ReplicateAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ReplicateAttrs, replicate_legion_dim, replicate_degree);

namespace std {
template <>
struct hash<::FlexFlow::ReplicateAttrs> {
  size_t operator()(::FlexFlow::ReplicateAttrs const &) const;
};
} 

#endif 
