#ifndef _FLEXFLOW_PARTITION_PARAMS_H
#define _FLEXFLOW_PARTITION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct RepartitionParams : public OpParamsInterface {
public:
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  int repartition_legion_dim;
  int repartition_degree;
};
bool operator==(RepartitionParams const &, RepartitionParams const &);
bool operator<(RepartitionParams const &, RepartitionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::RepartitionParams> {
  size_t operator()(FlexFlow::RepartitionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_PARTITION_PARAMS_H
