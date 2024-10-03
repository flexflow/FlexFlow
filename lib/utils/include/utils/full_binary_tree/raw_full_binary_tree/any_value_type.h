#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_RAW_FULL_BINARY_TREE_ANY_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_RAW_FULL_BINARY_TREE_ANY_VALUE_TYPE_H

#include <any>
#include <functional>
#include <string>
#include <fmt/format.h>

namespace FlexFlow {

struct any_value_type {
public:
  any_value_type(std::any const &value,
               std::function<bool(std::any const &, std::any const &)> const &eq,
               std::function<bool(std::any const &, std::any const &)> const &neq,
               std::function<size_t(std::any const &)> const &hash,
               std::function<std::string(std::any const &)> const &to_string);

  bool operator==(any_value_type const &other) const;
  bool operator!=(any_value_type const &other) const;

  template <typename T>
  T get() const {
    return std::any_cast<T>(value);
  }

  friend std::string format_as(any_value_type const &);
private:
  std::any value;
  std::function<bool(std::any const &, std::any const &)> eq;
  std::function<bool(std::any const &, std::any const &)> neq;
  std::function<size_t(std::any const &)> hash;
  std::function<std::string(std::any const &)> to_string;

  friend std::hash<any_value_type>;
};


template <typename T>
any_value_type make_any_value_type(T const &t) {
  return any_value_type{
    std::make_any<T>(t),
    [](std::any const &l, std::any const &r) {
      return std::any_cast<T>(l) == std::any_cast<T>(r);
    },
    [](std::any const &l, std::any const &r) {
      return std::any_cast<T>(l) != std::any_cast<T>(r);
    },
    [](std::any const &v) {
      return std::hash<T>{}(std::any_cast<T>(v));
    },
    [](std::any const &v) {
      return fmt::to_string(std::any_cast<T>(v));
    },
  };
}

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::any_value_type> {
  size_t operator()(::FlexFlow::any_value_type const &) const;
};

}

#endif
