#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/overload.h"

namespace FlexFlow {

bool training_op_attrs_has_op_type(TrainingOperationAttrs const &op_attrs,
                                   OperatorType op_type) {
  return op_attrs.visit<bool>(overload{
      [&](PCGOperatorAttrs const &pcg_op_attrs) -> bool {
        return pcg_op_attrs_get_op_type(pcg_op_attrs) == op_type;
      },
      [](LossAttrs const &) -> bool { return false; },
      [](CopyAttrs const &) -> bool { return false; },
      [](GradientReductionAttrs const &) -> bool { return false; },
  });
}

TrainingOpType training_op_attrs_get_op_type(
    TrainingOperationAttrs const &training_op_attrs) {
  return training_op_attrs.visit<TrainingOpType>(overload{
      [](PCGOperatorAttrs const &a) -> TrainingOpType {
        return TrainingOpType{
            pcg_op_attrs_get_op_type(a),
        };
      },
      [](LossAttrs const &) -> TrainingOpType {
        return TrainingOpType{TrainingOnlyOpType::LOSS};
      },
      [](CopyAttrs const &) -> TrainingOpType {
        return TrainingOpType{TrainingOnlyOpType::COPY};
      },
      [](GradientReductionAttrs const &) -> TrainingOpType {
        return TrainingOpType{TrainingOnlyOpType::GRADIENT_REDUCTION};
      },
  });
}

} // namespace FlexFlow
