#include "op-attrs/ops/loss_functions.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/overload.h"
#include <algorithm>
#include <cassert>

namespace FlexFlow {

LossFunction get_loss_function(LossAttrs const &attrs) {
  return attrs.visit<LossFunction>(
      overload{[&](SparseCategoricalCrossEntropyLossAttrs const &s) {
                 return LossFunction::SPARSE_CATEGORICAL_CROSSENTROPY;
               },
               [&](NonconfigurableLossAttrs const &s) { return s.loss_type; }});
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
    throw mk_runtime_error(fmt::format(
        "Unknown loss type {}. Please report this as an issue.", name));
  }
}

} // namespace FlexFlow
