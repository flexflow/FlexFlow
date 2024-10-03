#include "utils/full_binary_tree/raw_full_binary_tree/any_value_type.h"

namespace FlexFlow {

any_value_type::any_value_type(std::any const &value,
               std::function<bool(std::any const &, std::any const &)> const &eq,
               std::function<bool(std::any const &, std::any const &)> const &neq,
               std::function<size_t(std::any const &)> const &hash,
               std::function<std::string(std::any const &)> const &to_string)
  : value(value), eq(eq), neq(neq), hash(hash), to_string(to_string)
{}

bool any_value_type::operator==(any_value_type const &other) const {
  return this->eq(this->value, other.value);
}

bool any_value_type::operator!=(any_value_type const &other) const {
  return this->neq(this->value, other.value);
}

std::string format_as(any_value_type const &v) {
  return v.to_string(v.value);
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::any_value_type>::operator()(::FlexFlow::any_value_type const &v) const {
  return v.hash(v);
}

} // namespace std
