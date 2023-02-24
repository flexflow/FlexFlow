#ifndef _FLEXFLOW_COMBINE_PARAMS_H
#define _FLEXFLOW_COMBINE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct CombineParams : public UnaryOpParams {
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  ParallelTensorShape output_shape(ParallelTensorShape const &) const override;
  bool is_valid(ParallelTensorShape const &input) const override;
  OperatorType op_type() const override;
public:
  int combine_legion_dim;
  int combine_degree;
};
bool operator==(CombineParams const &, CombineParams const &);
bool operator<(CombineParams const &, CombineParams const &);

}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::CombineParams, combine_legion_dim, combine_degree);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::CombineParams> {
  size_t operator()(::FlexFlow::opmeta::CombineParams const &) const;
};
} 

#endif // _FLEXFLOW_COMBINE_PARAMS_H
