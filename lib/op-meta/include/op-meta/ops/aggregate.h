#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "visit_struct/visit_struct.hpp"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct AggregateAttrs : public UnaryOutputOpAttrs {
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
} // namespace std

#endif // _FLEXFLOW_AGGREGATE_PARAMS_H
