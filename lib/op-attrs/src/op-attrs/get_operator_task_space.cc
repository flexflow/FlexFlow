#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

OperatorTaskSpace get_operator_task_space(
    ComputationGraphOpAttrs const &attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &inputs_degrees) {
  return attrs.visit<OperatorTaskSpace>(overload{
      [&](ElementUnaryAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return element_unary_get_operator_task_space(attrs, input);
      },
      [&](ElementBinaryAttrs const &attrs) {
        auto [lhs, rhs] = require_two_keys(inputs_degrees,
                                           TensorSlotName::LHS_INPUT,
                                           TensorSlotName::RHS_INPUT);

        return element_binary_get_operator_task_space(
            /*attrs=*/attrs,
            /*lhs_input_degrees=*/lhs,
            /*rhs_input_degrees=*/rhs);
      },
      [&](LinearAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return linear_get_operator_task_space(attrs, input);
      },
      [&](InputAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 0);

        return input_get_operator_task_space(attrs);
      },
      [&](TransposeAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return transpose_get_operator_task_space(attrs, input);
      },
      [&](WeightAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 0);

        return weight_get_operator_task_space(attrs);
      },
      [](auto const &attrs) -> OperatorTaskSpace {
        PANIC("Missing implmentation of get_operator_task_space", attrs);
      },
  });
}

} // namespace FlexFlow
