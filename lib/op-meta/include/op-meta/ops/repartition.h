#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct RepartitionAttrs : public UnaryOpAttrs {
public:
  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int repartition_legion_dim;
  int repartition_degree;
};
bool operator==(RepartitionAttrs const &, RepartitionAttrs const &);
bool operator<(RepartitionAttrs const &, RepartitionAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::RepartitionAttrs, repartition_legion_dim, repartition_degree);

namespace std {
template <>
struct hash<::FlexFlow::RepartitionAttrs> {
  size_t operator()(::FlexFlow::RepartitionAttrs const &) const;
};
}

#endif 
