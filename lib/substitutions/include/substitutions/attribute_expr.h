#ifndef _FLEXFLOW_SUBSTITUTIONS_CONSTRAINT_H
#define _FLEXFLOW_SUBSTITUTIONS_CONSTRAINT_H

#include "mpark/variant.hpp"

namespace FlexFlow {

enum class ConstraintType { EQUAL };

template <typename T>
struct ListIndexAccess {
  T attribute_key;
  int index;
};

template <typename T>
struct ListSize {
  T attribute_key;
};

template <typename T>
using AttributeExpr = variant<T, ListIndexAccess<T>, ListSize<T>>;

template <typename K, typename V>
struct AttributeConstraint {
  ConstraintType constraint_type;
  AttributeExpr<K> attribute_expr;
  V attribute_value;
};

} // namespace FlexFlow

#endif
