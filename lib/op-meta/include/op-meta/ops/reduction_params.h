#ifndef _FLEXFLOW_REDUCTION_PARAMS_H
#define _FLEXFLOW_REDUCTION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct ReductionParams : public OpParamsInterface {
public:
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  bool is_valid(std::vector<ParallelTensorShape> const &) const override;
  int num_outputs(std::vector<ParallelTensorShape> const &) const override;
  OperatorType op_type() const override;
public:
  int reduction_legion_dim;
  int reduction_degree;
};
bool operator==(ReductionParams const &, ReductionParams const &);
bool operator<(ReductionParams const &, ReductionParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::ReductionParams> {
  size_t operator()(FlexFlow::ReductionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_REDUCTION_PARAMS_H
