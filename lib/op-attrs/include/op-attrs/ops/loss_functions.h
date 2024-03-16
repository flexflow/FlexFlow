#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H

#include "core.h"
#include "utils/exception.h"
#include "utils/visitable.h"
#include <variant>

namespace FlexFlow {

enum class LossFunction {
  CATEGORICAL_CROSSENTROPY,
  SPARSE_CATEGORICAL_CROSSENTROPY,
  MEAN_SQUARED_ERROR_AVG_REDUCE,
  MEAN_SQUARED_ERROR_SUM_REDUCE,
  IDENTITY
};

std::string format_as(LossFunction const &);
CHECK_FMTABLE(LossFunction);

LossFunction parse_loss_function_name(std::string const &);

struct SparseCategoricalCrossEntropyLossAttrs {
  req<bool> replace_labels; // for aggregate_spec: More predictions than labels
};
FF_VISITABLE_STRUCT(SparseCategoricalCrossEntropyLossAttrs, replace_labels);
FF_VISIT_FMTABLE(SparseCategoricalCrossEntropyLossAttrs);
CHECK_VALID_OP_ATTR(SparseCategoricalCrossEntropyLossAttrs);

struct OtherLossAttrs {
  req<LossFunction> loss_type;
};
FF_VISITABLE_STRUCT(OtherLossAttrs, loss_type);
FF_VISIT_FMTABLE(OtherLossAttrs);
CHECK_VALID_OP_ATTR(OtherLossAttrs);

using LossAttrs =
    std::variant<SparseCategoricalCrossEntropyLossAttrs, OtherLossAttrs>;

LossFunction get_loss_function(OtherLossAttrs const &);
LossFunction get_loss_function(SparseCategoricalCrossEntropyLossAttrs const &);
LossFunction get_loss_function(LossAttrs const &);

} // namespace FlexFlow

#endif
