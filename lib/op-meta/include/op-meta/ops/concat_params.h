#ifndef _FLEXFLOW_CONCAT_PARAMS_H
#define _FLEXFLOW_CONCAT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct ConcatParams : public OpParamsInterface {
public:
  using AsConstTuple = std::tuple<int>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &) const override;
  OperatorType op_type() const override;
public:
  int axis;

};

bool operator==(ConcatParams const &, ConcatParams const &);
bool operator<(ConcatParams const &, ConcatParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ConcatParams> {
  size_t operator()(FlexFlow::ConcatParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_CONCAT_PARAMS_H
