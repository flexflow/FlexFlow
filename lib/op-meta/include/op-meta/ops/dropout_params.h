#ifndef _FLEXFLOW_DROPOUT_PARAMS_H
#define _FLEXFLOW_DROPOUT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct DropoutParams : public UnaryOpParams {
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  float rate;
  unsigned long long seed;
};

bool operator==(DropoutParams const &, DropoutParams const &);
bool operator<(DropoutParams const &, DropoutParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::DropoutParams, rate, seed);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::DropoutParams> {
  size_t operator()(::FlexFlow::opmeta::DropoutParams const &) const;
};
} 

#endif // _FLEXFLOW_DROPOUT_PARAMS_H
