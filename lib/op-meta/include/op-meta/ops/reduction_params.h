#ifndef _FLEXFLOW_REDUCTION_PARAMS_H
#define _FLEXFLOW_REDUCTION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ReductionParams : public UnaryOpParams {
public:
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  ParallelTensorShape output_shape(ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  int reduction_legion_dim;
  int reduction_degree;
};
bool operator==(ReductionParams const &, ReductionParams const &);
bool operator<(ReductionParams const &, ReductionParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ReductionParams, reduction_legion_dim, reduction_degree);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ReductionParams> {
  size_t operator()(::FlexFlow::opmeta::ReductionParams const &) const;
};
} 

#endif // _FLEXFLOW_REDUCTION_PARAMS_H
