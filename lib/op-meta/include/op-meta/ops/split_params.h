#ifndef _FLEXFLOW_SPLIT_PARAMS_H
#define _FLEXFLOW_SPLIT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"
#include "op-meta/visit_struct.h"

namespace FlexFlow {
namespace opmeta {

struct SplitParams : public OpParamsInterface {
public:
  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override; 
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &input_shapes) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> splits;
  int legion_axis;
};

bool operator==(SplitParams const &, SplitParams const &);
bool operator<(SplitParams const &, SplitParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::SplitParams, splits, legion_axis);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::SplitParams> {
  size_t operator()(::FlexFlow::opmeta::SplitParams const &) const;
};
} 

#endif // _FLEXFLOW_SPLIT_PARAMS_H
