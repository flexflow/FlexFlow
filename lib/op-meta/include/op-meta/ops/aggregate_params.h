#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "visit_struct/visit_struct.hpp"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {
namespace opmeta {

struct AggregateParams : public UnaryOutput {
  bool is_valid(std::vector<ParallelTensorShape> const &) const override;
  ParallelTensorShape output_shape(std::vector<ParallelTensorShape> const &input_shapes) const override;
  OperatorType op_type() const override;
public:
  int n;
  float lambda_bal;
};

bool operator==(AggregateParams const &, AggregateParams const &);
bool operator<(AggregateParams const &, AggregateParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::AggregateParams, n, lambda_bal);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::AggregateParams> {
  size_t operator()(::FlexFlow::opmeta::AggregateParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_AGGREGATE_PARAMS_H
