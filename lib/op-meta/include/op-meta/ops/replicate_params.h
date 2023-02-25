#ifndef _FLEXFLOW_REPLICATE_PARAMS_H
#define _FLEXFLOW_REPLICATE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ReplicateParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  int replicate_legion_dim;
  int replicate_degree;
};

bool operator==(ReplicateParams const &, ReplicateParams const &);
bool operator<(ReplicateParams const &, ReplicateParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ReplicateParams, replicate_legion_dim, replicate_degree);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ReplicateParams> {
  size_t operator()(::FlexFlow::opmeta::ReplicateParams const &) const;
};
} 

#endif // _FLEXFLOW_REPLICATE_PARAMS_H
