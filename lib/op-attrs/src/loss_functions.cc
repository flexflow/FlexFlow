#include "op-attrs/ops/loss_functions.h"
#include "utils/containers.h"
#include <algorithm>
#include <cassert>

namespace FlexFlow {

SparseCategoricalCrossEntropyLossAttrs::SparseCategoricalCrossEntropyLossAttrs(
    bool _replace_labels)
    : replace_labels(_replace_labels) {}

OtherLossAttrs::OtherLossAttrs(LossFunction _loss_type)
    : loss_type(_loss_type) {
  assert(this->loss_type != LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY);
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
