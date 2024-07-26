#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_PAIR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_PAIR_H

#include "utils/fmt/pair.h"
#include "utils/hash-utils.h"
#include "utils/type_traits_core.h"
#include <rapidcheck.h>
#include <tuple>

namespace FlexFlow {

template <typename T>
struct commutative_pair {
public:
  commutative_pair() = delete;
  commutative_pair(T const &x, T const &y) : first(x), second(y) {}

  bool operator==(commutative_pair const &other) const {
    return this->tie() == other.tie() || this->rtie() == other.tie();
  }

  bool operator!=(commutative_pair const &other) const {
    return this->tie() != other.tie() && this->rtie() != other.tie();
  }

  bool operator<(commutative_pair const &other) const {
    static_assert(is_lt_comparable_v<T>);

    return this->otie() < other.otie();
  }

  bool operator>(commutative_pair const &other) const {
    static_assert(is_lt_comparable_v<T>);

    return this->otie() > other.otie();
  }

  bool operator<=(commutative_pair const &other) const {
    static_assert(is_lt_comparable_v<T>);

    return this->otie() <= other.otie();
  }

  bool operator>=(commutative_pair const &other) const {
    static_assert(is_lt_comparable_v<T>);

    return this->otie() >= other.otie();
  }

  T const &max() const {
    static_assert(is_lt_comparable_v<T>);
    return std::max(this->first, this->second);
  }

  T const &min() const {
    static_assert(is_lt_comparable_v<T>);
    return std::min(this->first, this->second);
  }

  std::pair<T, T> ordered() const {
    return std::make_pair(this->first, this->second);
  }

private:
  T first;
  T second;

private:
  std::tuple<T const &, T const &> tie() const {
    return std::tie(this->first, this->second);
  }
  std::tuple<T const &, T const &> rtie() const {
    return std::tie(this->second, this->first);
  }

  std::tuple<T const &, T const &> otie() const {
    return std::tie(this->max(), this->min());
  }

  friend ::std::hash<commutative_pair<T>>;
};

template <typename T>
std::pair<T, T> format_as(commutative_pair<T> const &p) {
  return p.ordered();
}

template <typename T>
std::ostream &operator<<(std::ostream &s, commutative_pair<T> const &p) {
  return (s << fmt::to_string(p));
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::commutative_pair<T>> {
  size_t operator()(::FlexFlow::commutative_pair<T> const &p) {
    size_t result = 0;
    ::FlexFlow::unordered_hash_combine(result, p.first);
    ::FlexFlow::unordered_hash_combine(result, p.second);
    return result;
  }
};

} // namespace std

namespace rc {

template <typename T>
struct Arbitrary<::FlexFlow::commutative_pair<T>> {
  static Gen<::FlexFlow::commutative_pair<T>> arbitrary() {
    return gen::map<std::pair<T, T>>([](std::pair<T, T> const &p) {
      return ::FlexFlow::commutative_pair{p.first, p.second};
    });
  }
};

} // namespace rc

#endif
