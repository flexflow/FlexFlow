#ifndef _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H
#define _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H

#include "stack_vector.h"
#include "utils/fmt.h"
#include <cstring>
#include <string>

namespace FlexFlow {

template <typename Char, size_t MAXSIZE>
struct stack_basic_string {
  stack_basic_string() = default;

  stack_basic_string(Char const *c) : contents(c, c + std::strlen(c)) {}

  stack_basic_string(std::basic_string<Char> const &s)
      : stack_basic_string(s.c_str()) {}

  operator std::basic_string<Char>() const {
    std::basic_string<Char> result;
    for (Char const &c : this->contents) {
      result.push_back(c);
    }
    return result;
  }

  std::size_t size() const {
    return this->contents.size();
  }

  std::size_t length() const {
    return this->size();
  }

  friend bool operator==(stack_basic_string const &lhs,
                         stack_basic_string const &rhs) {
    return lhs.contents == rhs.contents;
  }

  friend bool operator!=(stack_basic_string const &lhs,
                         stack_basic_string const &rhs) {
    return lhs.contents != rhs.contents;
  }

  friend bool operator<(stack_basic_string const &lhs,
                        stack_basic_string const &rhs) {
    return lhs.contents < rhs.contents;
  }

  friend struct std::hash<stack_basic_string>;

private:
  stack_vector<Char, MAXSIZE> contents;
};

template <size_t MAXSIZE>
using stack_string = stack_basic_string<char, MAXSIZE>;

} // namespace FlexFlow

namespace std {

template <typename Char, size_t MAXSIZE>
struct hash<::FlexFlow::stack_basic_string<Char, MAXSIZE>> {
  size_t
      operator()(::FlexFlow::stack_basic_string<Char, MAXSIZE> const &s) const {
    return get_std_hash(s.contents);
  }
};

} // namespace std

namespace fmt {

template <int MAXSIZE>
struct formatter<::FlexFlow::stack_string<MAXSIZE>> : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::stack_string<MAXSIZE> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    return formatter<std::string>::format(static_cast<std::string>(m), ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

static_assert(is_default_constructible<stack_string<1>>::value,
              "stack_string must be default constructible");
static_assert(is_copy_constructible<stack_string<1>>::value,
              "stack_string must be copy constructible");
static_assert(is_move_constructible<stack_string<1>>::value,
              "stack_string must be move constructible");
static_assert(is_copy_assignable<stack_string<1>>::value,
              "stack_string must be copy assignable");
static_assert(is_move_assignable<stack_string<1>>::value,
              "stack_string must be move assignable");
static_assert(is_equal_comparable<stack_string<1>>::value,
              "stack_string must support ==");
static_assert(is_neq_comparable<stack_string<1>>::value,
              "stack_string must support !=");
static_assert(is_lt_comparable<stack_string<1>>::value,
              "stack_string must support <");
static_assert(is_hashable<stack_string<1>>::value,
              "stack_string must be hashable");

} // namespace FlexFlow

#endif
