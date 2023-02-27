#ifndef _FLEXFLOW_OP_META_OPS_REDUCE_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_REDUCE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include <vector>
#include "unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ReduceParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
  bool is_valid(ParallelTensorShape const &) const override;
public:
  std::vector<int> axes;
  bool keepdims;
};

bool operator==(ReduceParams const &, ReduceParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ReduceParams, axes, keepdims);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ReduceParams> {
  size_t operator()(::FlexFlow::opmeta::ReduceParams const &) const;
};
}

#endif
