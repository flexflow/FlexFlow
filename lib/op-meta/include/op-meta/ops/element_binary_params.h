#ifndef _FLEXFLOW_ELEMENT_BINARY_PARAMS_H
#define _FLEXFLOW_ELEMENT_BINARY_PARAMS_H

#include "op-meta/ffconst.h"
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/binary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ElementBinaryParams : public BinaryOpParams {
public:
  bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  OperatorType type;
};

bool operator==(ElementBinaryParams const &, ElementBinaryParams const &);
bool operator<(ElementBinaryParams const &, ElementBinaryParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ElementBinaryParams, type);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ElementBinaryParams> {
  size_t operator()(::FlexFlow::opmeta::ElementBinaryParams const &) const;
};
}

#endif // _FLEXFLOW_ELEMENT_BINARY_PARAMS_H
