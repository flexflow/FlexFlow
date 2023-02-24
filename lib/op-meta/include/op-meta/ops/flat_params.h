#ifndef _FLEXFLOW_FLAT_PARAMS_H
#define _FLEXFLOW_FLAT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct FlatParams : public UnaryOpParams {
  bool is_valid(ParallelTensorShape const &input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
private:
  int output_size(ParallelTensorShape const &input,
                  ParallelTensorShape &output) const;

  ParallelTensorShape calculate_output_shape(ParallelTensorShape const &input) const;
};

bool operator==(FlatParams const &, FlatParams const &);
bool operator<(FlatParams const &, FlatParams const &);

}
}

namespace std {
template <>
struct hash<::FlexFlow::opmeta::FlatParams> {
  size_t operator()(::FlexFlow::opmeta::FlatParams const &) const;
};
}

#endif // _FLEXFLOW_FLAT_PARAMS_H
