#ifndef _FLEXFLOW_GROUPBY_PARAMS_H
#define _FLEXFLOW_GROUPBY_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "binary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct Group_byParams : public BinaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  int n;
  float alpha;
};
bool operator==(Group_byParams const &, Group_byParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::Group_byParams, n, alpha);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::Group_byParams> {
  size_t operator()(::FlexFlow::opmeta::Group_byParams const &) const;
};
}

#endif 
