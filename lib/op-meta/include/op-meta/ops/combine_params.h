#ifndef _FLEXFLOW_COMBINE_PARAMS_H
#define _FLEXFLOW_COMBINE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct CombineParams : public OpParamsInterface {
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  int combine_legion_dim;
  int combine_degree;
};
bool operator==(CombineParams const &, CombineParams const &);
bool operator<(CombineParams const &, CombineParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CombineParams> {
  size_t operator()(FlexFlow::CombineParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_COMBINE_PARAMS_H
