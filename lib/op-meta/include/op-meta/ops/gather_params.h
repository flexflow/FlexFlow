#ifndef _FLEXFLOW_GATHER_PARAMS_H
#define _FLEXFLOW_GATHER_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "visit_struct/visit_struct.hpp"
#include "binary_op.h"

namespace FlexFlow {
namespace opmeta {

struct GatherParams : public BinaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
  bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const override;
public:
  int legion_dim;
};

bool operator==(GatherParams const &, GatherParams const &);
bool operator<(GatherParams const &, GatherParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::GatherParams, legion_dim);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::GatherParams> {
  size_t operator()(::FlexFlow::opmeta::GatherParams const &) const;
};
} 

#endif 
