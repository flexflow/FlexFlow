#ifndef _FLEXFLOW_CONCAT_PARAMS_H
#define _FLEXFLOW_CONCAT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct ConcatParams : public OpParamsInterface {
public:
  bool is_valid(std::vector<ParallelTensorShape> const &input_shapes) const override;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &input_shapes) const override;
  OperatorType op_type() const override;
public:
  int axis;

};

bool operator==(ConcatParams const &, ConcatParams const &);
bool operator<(ConcatParams const &, ConcatParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::ConcatParams, axis);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::ConcatParams> {
  size_t operator()(::FlexFlow::opmeta::ConcatParams const &) const;
};
} 

#endif // _FLEXFLOW_CONCAT_PARAMS_H
