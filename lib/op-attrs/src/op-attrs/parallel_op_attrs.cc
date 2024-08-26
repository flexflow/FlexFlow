#include "op-attrs/parallel_op_attrs.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "utils/overload.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ParallelOpAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  return attrs.visit<ParallelTensorShape>(overload{
      [&](CombineAttrs const &combine_attrs) {
        return throw_if_unexpected(
            get_output_shape(combine_attrs, input_shape));
      },
      [&](ReductionAttrs const &reduction_attrs) {
        return throw_if_unexpected(
            get_output_shape(reduction_attrs, input_shape));
      },
      [&](RepartitionAttrs const &repartition_attrs) {
        return throw_if_unexpected(
            get_output_shape(repartition_attrs, input_shape));
      },
      [&](ReplicateAttrs const &replicate_attrs) {
        return get_output_shape(replicate_attrs, input_shape);
      },
  });
}

PCGOperatorAttrs
    pcg_op_attrs_from_parallel_op_attrs(ParallelOpAttrs const &attrs) {
  return attrs.visit<PCGOperatorAttrs>(
      [](auto const &attrs) { return PCGOperatorAttrs{attrs}; });
}

} // namespace FlexFlow
