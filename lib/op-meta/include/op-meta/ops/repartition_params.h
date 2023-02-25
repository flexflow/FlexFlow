#ifndef _FLEXFLOW_PARTITION_PARAMS_H
#define _FLEXFLOW_PARTITION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct RepartitionParams : public UnaryOpParams {
public:
  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int repartition_legion_dim;
  int repartition_degree;
};
bool operator==(RepartitionParams const &, RepartitionParams const &);
bool operator<(RepartitionParams const &, RepartitionParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::RepartitionParams, repartition_legion_dim, repartition_degree);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::RepartitionParams> {
  size_t operator()(::FlexFlow::opmeta::RepartitionParams const &) const;
};
}

#endif // _FLEXFLOW_PARTITION_PARAMS_H
