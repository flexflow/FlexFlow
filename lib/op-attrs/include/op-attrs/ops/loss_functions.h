#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H

#include "core.h"
#include "utils/exception.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class LossFunction {
  CATEGORICAL_CROSSENTROPY,
  SPARSE_CATEGORICAL_CROSSENTROPY,
  MEAN_SQUARED_ERROR_AVG_REDUCE,
  MEAN_SQUARED_ERROR_SUM_REDUCE,
  IDENTITY
};

LossFunction parse_loss_function_name(std::string const &);

struct SparseCategoricalCrossEntropyLossAttrs {
  req<bool> replace_labels; // for aggregate_spec: More predictions than labels
};
FF_VISITABLE_STRUCT(SparseCategoricalCrossEntropyLossAttrs, replace_labels);
CHECK_VALID_OP_ATTR(SparseCategoricalCrossEntropyLossAttrs);

struct OtherLossAttrs {
  req<LossFunction> loss_type;
};
FF_VISITABLE_STRUCT(OtherLossAttrs, loss_type);
CHECK_VALID_OP_ATTR(OtherLossAttrs);

using LossAttrs =
    variant<SparseCategoricalCrossEntropyLossAttrs, OtherLossAttrs>;

LossFunction get_loss_function(OtherLossAttrs const &);
LossFunction get_loss_function(SparseCategoricalCrossEntropyLossAttrs const &);
LossFunction get_loss_function(LossAttrs const &);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::LossFunction> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::LossFunction d, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (d) {
      case LossFunction::CATEGORICAL_CROSSENTROPY:
        name = "CategoricalCrossEntropy";
        break;
      case LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY:
        name = "SparseCategoricalCrossEntropy";
        break;
      case LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE:
        name = "MeanSquaredErrorAvgReduce";
        break;
      case LossFunction::MEAN_SQUARED_ERROR_SUM_REDUCE:
        name = "MeanSquaredErrorSumReduce";
        break;
      case LossFunction::IDENTITY:
        name = "Identity";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
