#ifndef _FLEXFLOW_SPLIT_PARAMS_H
#define _FLEXFLOW_SPLIT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct SplitParams : public OpParamsInterface {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<std::vector<int>, int>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override; 
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> splits;
  int legion_axis;
};

bool operator==(SplitParams const &, SplitParams const &);
bool operator<(SplitParams const &, SplitParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SplitParams> {
  size_t operator()(FlexFlow::SplitParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SPLIT_PARAMS_H
