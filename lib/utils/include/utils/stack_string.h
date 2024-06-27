#ifndef _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H
#define _FLEXFLOW_UTILS_INCLUDE_STACK_STRING_H

#include "fmt/core.h"
#include "stack_vector.h"
#include "utils/fmt.h"
#include "utils/json.h"
#include "utils/type_traits.h"
#include <cstring>
#include <rapidcheck.h>
#include <string>

namespace FlexFlow {

template <typename Char, size_t MAXSIZE>
struct stack_basic_string {
  stack_basic_string() = default;

  stack_basic_string(Char const *c) : contents(c, c + std::strlen(c)) {}

  template <typename Iterator>
  stack_basic_string(Iterator start, Iterator end) : contents(start, end) {}

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

  friend fmt::basic_string_view<Char>
      format_as(stack_basic_string<Char, MAXSIZE> const &s) {
    return {s.contents.data(), s.length()};
  }

private:
  stack_vector<Char, MAXSIZE> contents;
};

template <size_t MAXSIZE>
using stack_string = stack_basic_string<char, MAXSIZE>;

template <std::size_t MAXSIZE>
void to_json(json &j, stack_string<MAXSIZE> const &v) {
  std::string as_string = v;
  j = as_string;
}

template <std::size_t MAXSIZE>
void from_json(json const &j, stack_string<MAXSIZE> &v) {
  std::string as_string;
  j.get_to(as_string);
  v = stack_string<MAXSIZE>{as_string};
}

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

namespace rc {

template <typename Char, size_t MAXSIZE>
struct Arbitrary<::FlexFlow::stack_basic_string<Char, MAXSIZE>> {
  static Gen<::FlexFlow::stack_basic_string<Char, MAXSIZE>> arbitrary() {
    return gen::mapcat(gen::inRange<size_t>(0, MAXSIZE), [](size_t size) {
      return gen::container<::FlexFlow::stack_basic_string<Char, MAXSIZE>>(
          size, gen::arbitrary<Char>());
    });
  }
};

} // namespace rc

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
CHECK_WELL_BEHAVED_VALUE_TYPE(stack_string<1>);
CHECK_HASHABLE(stack_string<1>);
// CHECK_FMTABLE(stack_string<1>);

} // namespace FlexFlow

#endif
