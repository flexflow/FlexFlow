#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"

namespace FlexFlow {

TensorAttributePattern tensor_attribute_pattern_match_all() {
  return TensorAttributePattern{{}};
}

} // namespace FlexFlow
