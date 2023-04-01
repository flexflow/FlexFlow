#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/ops/unary_op.h"

namespace FlexFlow {

struct AggregateAttrs : public UnaryOutputOpAttrs {
  AggregateAttrs() = delete;
  AggregateAttrs(int n, float lambda_bal);

  bool is_valid(std::vector<ParallelTensorShape> const &) const override;
  ParallelTensorShape output_shape(std::vector<ParallelTensorShape> const &input_shapes) const override;
  OperatorType op_type() const override;
public:
  int n;
  float lambda_bal;
};

bool operator==(AggregateAttrs const &, AggregateAttrs const &);
bool operator<(AggregateAttrs const &, AggregateAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::AggregateAttrs, n, lambda_bal);

namespace std {
template <>
struct hash<::FlexFlow::AggregateAttrs> {
  size_t operator()(::FlexFlow::AggregateAttrs const &) const;
};
}

#endif 
