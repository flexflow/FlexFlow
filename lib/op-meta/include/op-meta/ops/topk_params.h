#ifndef _FLEXFLOW_TOPK_PARAMS_H
#define _FLEXFLOW_TOPK_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct TopKParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;

public:
  int k;
  bool sorted;
};
bool operator==(TopKParams const &, TopKParams const &);
bool operator<(TopKParams const &, TopKParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::TopKParams, k, sorted);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::TopKParams> {
  size_t operator()(::FlexFlow::opmeta::TopKParams const &) const;
};
}

#endif // _FLEXFLOW_TOPK_PARAMS_H
