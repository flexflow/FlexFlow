#include "substitutions/tensor_pattern/eval_list_size.h"
#include "substitutions/tensor_pattern/get_attribute.h"
#include "utils/overload.h"

namespace FlexFlow {

TensorAttributeValue eval_list_size(ParallelTensorAttrs const &attrs, TensorAttributeListSize const &acc) {
  TensorAttributeValue from_attr = get_attribute(attrs, acc.attribute_key);

  return from_attr.visit<TensorAttributeValue>(overload {
    [](std::vector<int> const &v) -> TensorAttributeValue { 
      return TensorAttributeValue{v.size()}; 
    },
    [](auto &&) -> TensorAttributeValue { throw mk_runtime_error("Invalid operand"); },
  });
}

} // namespace FlexFlow
