#ifndef _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H
#define _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct AggregateSpecAttrs : public UnaryOpAttrs {
public:
  AggregateSpecAttrs() = delete;
  AggregateSpecAttrs(int n, float lambda_bal);

  OperatorType op_type() const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
public:
  int n;
  float lambda_bal;
};
bool operator==(AggregateSpecAttrs const &, AggregateSpecAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::AggregateSpecAttrs, n, lambda_bal);

namespace std {
template <>
struct hash<::FlexFlow::AggregateSpecAttrs> {
  size_t operator()(::FlexFlow::AggregateSpecAttrs const &) const;
};
}

#endif 
