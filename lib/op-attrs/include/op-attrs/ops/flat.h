#ifndef _FLEXFLOW_FLAT_ATTRS_H
#define _FLEXFLOW_FLAT_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct FlatAttrs : public UnaryOpAttrs {
  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
private:
  int output_size(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input) const;
};

bool operator==(FlatAttrs const &, FlatAttrs const &);
bool operator<(FlatAttrs const &, FlatAttrs const &);

}

namespace std {
template <>
struct hash<::FlexFlow::FlatAttrs> {
  size_t operator()(::FlexFlow::FlatAttrs const &) const;
};
}

#endif 
