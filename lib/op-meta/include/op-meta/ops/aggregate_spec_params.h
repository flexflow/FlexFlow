#ifndef _FLEXFLOW_AGGREGATE_SPEC_PARAMS_H
#define _FLEXFLOW_AGGREGATE_SPEC_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct AggregateSpecParams : public UnaryOpParams {
public:
  OperatorType op_type() const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
public:
  int n;
  float lambda_bal;
};
bool operator==(AggregateSpecParams const &, AggregateSpecParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::AggregateSpecParams, n, lambda_bal);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::AggregateSpecParams> {
  size_t operator()(::FlexFlow::opmeta::AggregateSpecParams const &) const;
};
}

#endif // _FLEXFLOW_AGGREGATE_SPEC_PARAMS_H
