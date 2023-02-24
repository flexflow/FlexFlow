#ifndef _FLEXFLOW_CAST_PARAMS_H
#define _FLEXFLOW_CAST_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct CastParams : public UnaryOpParams {
  bool is_valid(ParallelTensorShape const &) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  DataType dtype;
};

bool operator==(CastParams const &, CastParams const &);
bool operator<(CastParams const &, CastParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::CastParams, dtype);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::CastParams> {
  size_t operator()(::FlexFlow::opmeta::CastParams const &) const;
};
} 

#endif // _FLEXFLOW_CAST_PARAMS_H
