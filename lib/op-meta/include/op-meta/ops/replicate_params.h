#ifndef _FLEXFLOW_REPLICATE_PARAMS_H
#define _FLEXFLOW_REPLICATE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct ReplicateParams : public OpParamsInterface {
public:
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  int replicate_legion_dim;
  int replicate_degree;
};

bool operator==(ReplicateParams const &, ReplicateParams const &);
bool operator<(ReplicateParams const &, ReplicateParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReplicateParams> {
  size_t operator()(FlexFlow::ReplicateParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_REPLICATE_PARAMS_H
