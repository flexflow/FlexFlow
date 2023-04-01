#ifndef _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include <vector>
#include "unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReduceAttrs {
/* public: */
/*   ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override; */
/*   OperatorType op_type() const override; */
/*   bool is_valid(ParallelTensorShape const &) const override; */
public:
  std::vector<int> axes;
  bool keepdims;
};

bool operator==(ReduceAttrs const &, ReduceAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::ReduceAttrs, axes, keepdims);

namespace std {
template <>
struct hash<::FlexFlow::ReduceAttrs> {
  size_t operator()(::FlexFlow::ReduceAttrs const &) const;
};
}

#endif
