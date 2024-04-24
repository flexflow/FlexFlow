#include "substitutions/tensor_pattern/tensor_attribute_expr.h"
#include "substitutions/tensor_pattern/get_attribute.h"
#include "substitutions/tensor_pattern/eval_list_size.h"
#include "substitutions/tensor_pattern/eval_list_access.h"
#include "utils/overload.h"

namespace FlexFlow {

TensorAttributeValue
    evaluate_attribute_expr(ParallelTensorAttrs const &attrs,
                            TensorAttributeExpr const &expr) {

  return expr.visit<TensorAttributeValue>(overload {
    [&](TensorAttributeKey const &key) { 
      return get_attribute(attrs, key); 
    },
    [&](TensorAttributeListSize const &s) {
      return eval_list_size(attrs, s);
    },
    [&](TensorAttributeListIndexAccess const &s) {
      return eval_list_access(attrs, s);
    }
  });
}

} // namespace FlexFlow
