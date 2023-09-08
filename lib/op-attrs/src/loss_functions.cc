#include "op-attrs/ops/loss_functions.h"
#include "utils/containers.h"
#include <algorithm>
#include <cassert>

namespace FlexFlow {

std::string format_as(LossFunction const &d) {
  switch (d) {
    case LossFunction::CATEGORICAL_CROSSENTROPY:
      return "CategoricalCrossEntropy";
    case LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY:
      return "SparseCategoricalCrossEntropy";
    case LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE:
      return "MeanSquaredErrorAvgReduce";
    case LossFunction::MEAN_SQUARED_ERROR_SUM_REDUCE:
      return "MeanSquaredErrorSumReduce";
    case LossFunction::IDENTITY:
      return "Identity";
    default:
      throw mk_runtime_error("Unknown LossFunction with value {}", static_cast<int>(d));
  }
}

LossFunction get_loss_type(OtherLossAttrs const &attrs) {
  return attrs.loss_type;
}
LossFunction
    get_loss_type(SparseCategoricalCrossEntropyLossAttrs const &attrs) {
  return LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY;
}

struct GetLossFunction {
  template <typename T>
  LossFunction operator()(T const &t) {
    return get_loss_type(t);
  }
};

LossFunction get_loss_type(LossAttrs const &attrs) {
  return visit(GetLossFunction{}, attrs);
}

LossFunction parse_loss_name(std::string const &raw_name) {
  std::string name =
      transform(raw_name, [](unsigned char c) { return std::tolower(c); });

  if (name == "categorical_crossentropy") {
    return LossFunction::CATEGORICAL_CROSSENTROPY;
  } else if (name == "sparse_categorical_crossentropy") {
    return LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY;
  } else if (name == "mean_squared_error") {
    return LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE;
  } else if (name == "identity") {
    return LossFunction::IDENTITY;
  } else {
    throw mk_runtime_error(
        "Unknown loss type {}. Please report this as an issue.", name);
  }
}



} // namespace FlexFlow
