#include "substitutions/tensor_pattern/eval_list_size.h"
#include "substitutions/tensor_pattern/get_attribute.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/overload.h"

namespace FlexFlow {

TensorAttributeValue eval_list_size(ParallelTensorAttrs const &attrs,
                                    TensorAttributeListSize const &acc) {
  TensorAttributeValue from_attr = get_attribute(attrs, acc.attribute_key);

  return from_attr.visit<TensorAttributeValue>(overload{
      [](std::vector<nonnegative_int> const &v) -> TensorAttributeValue {
        return TensorAttributeValue{num_elements(v)};
      },
      [](auto &&) -> TensorAttributeValue { PANIC("Invalid operand"); },
  });
}

} // namespace FlexFlow
