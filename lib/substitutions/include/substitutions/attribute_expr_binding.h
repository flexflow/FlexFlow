#ifndef _FLEXFLOW_SUBSTITUTIONS_ATTRIBUTE_EXPR_BINDING_H
#define _FLEXFLOW_SUBSTITUTIONS_ATTRIBUTE_EXPR_BINDING_H

#include "attribute_expr.h"

namespace FlexFlow {

struct attr_expr_id : public strong_typedef<attr_expr_id, std::string> {
  using strong_typedef::strong_typedef;
};

template <typename T>
struct AttributeExprBinding {
  void add_expr(attr_expr_id const &id, AttributeExpr<T> const &expr) {
    binding.emplace(id, expr);
  }

  GraphAttributeExpr<T> get_expr(attr_expr_id const &id) const {
    return binding.at(id);
  }
private:
  std::unordered_map<attr_expr_id, AttributeExpr<T>> binding;
};

}

#endif
