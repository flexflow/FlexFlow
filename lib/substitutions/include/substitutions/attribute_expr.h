#ifndef _FLEXFLOW_SUBSTITUTIONS_CONSTRAINT_H
#define _FLEXFLOW_SUBSTITUTIONS_CONSTRAINT_H

#include "utils/variant.h"

namespace FlexFlow {

enum class ConstraintType { EQUAL };

template <typename T>
struct ListIndexAccess {
  T attribute_key;
  req<int> index;
};

template <typename T>
struct ListSize {
  req<T> attribute_key;
};

template <typename T>
using AttributeExpr = std::variant<T, ListIndexAccess<T>, ListSize<T>>;

template <typename K, typename V>
struct AttributeConstraint {
  ConstraintType constraint_type;
  AttributeExpr<K> attribute_expr;
  V attribute_value;
};

template <typename K, typename V>
struct AttributePattern {
  std::vector<AttributeConstraint<K, V>> attribute_constraints;
  // TODO: Revert to unordered_set once we have visitable for templates
  // std::unordered_set<AttributeConstraint<K, V>> attribute_constraints;
};

} // namespace FlexFlow

#endif
