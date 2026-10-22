#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONEMPTY_UNORDERED_SET_NONEMPTY_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONEMPTY_UNORDERED_SET_NONEMPTY_UNORDERED_SET_H

#include "utils/fmt/unordered_set.h"
#include "utils/hash-utils.h"
#include "utils/hash/unordered_set.h"
#include "utils/positive_int/positive_int.h"
#include <libassert/assert.hpp>
#include <set>

namespace FlexFlow {

template <typename T>
struct nonempty_unordered_set {
public:
  nonempty_unordered_set() = delete;

  nonempty_unordered_set(std::initializer_list<T> const &vs) : raw(vs) {
    ASSERT(this->raw.size() > 0);
  }

  explicit nonempty_unordered_set(std::unordered_set<T> const &s) : raw(s) {
    ASSERT(this->raw.size() > 0);
  }

  bool operator==(nonempty_unordered_set const &other) const {
    return this->unwrap_as_unordered_set() == other.unwrap_as_unordered_set();
  }

  bool operator!=(nonempty_unordered_set const &other) const {
    return this->unwrap_as_unordered_set() != other.unwrap_as_unordered_set();
  }

  bool operator==(std::unordered_set<T> const &other) const {
    return this->unwrap_as_unordered_set() == other;
  }

  bool operator!=(std::unordered_set<T> const &other) const {
    return this->unwrap_as_unordered_set() != other;
  }

  void insert(T const &t) {
    this->raw.insert(t);
  }

  size_t size() const {
    return this->raw.size();
  };

  positive_int num_elements() const {
    return positive_int{this->raw.size()};
  };

  std::unordered_set<T> const &unwrap_as_unordered_set() const {
    return this->raw;
  }

  using value_type = T;

  typename std::unordered_set<T>::const_iterator begin() const {
    return this->raw.cbegin();
  }

  typename std::unordered_set<T>::const_iterator cbegin() const {
    return this->raw.cbegin();
  }

  typename std::unordered_set<T>::const_iterator end() const {
    return this->raw.cend();
  }

  typename std::unordered_set<T>::const_iterator cend() const {
    return this->raw.cend();
  }

private:
  std::unordered_set<T> raw;
};

template <typename T>
bool operator==(std::unordered_set<T> const &lhs,
                nonempty_unordered_set<T> const &rhs) {
  return lhs == rhs.unwrap_as_unordered_set();
}

template <typename T>
bool operator!=(std::unordered_set<T> const &lhs,
                nonempty_unordered_set<T> const &rhs) {
  return lhs != rhs.unwrap_as_unordered_set();
}

template <typename T>
std::unordered_set<T> format_as(nonempty_unordered_set<T> const &s) {
  return s.unwrap_as_unordered_set();
}

template <typename T>
std::ostream &operator<<(std::ostream &s, nonempty_unordered_set<T> const &m) {
  return (s << fmt::to_string(m));
}

} // namespace FlexFlow

namespace rc {

template <typename T>
struct Arbitrary<::FlexFlow::nonempty_unordered_set<T>> {
  static Gen<::FlexFlow::nonempty_unordered_set<T>> arbitrary() {
    return gen::construct<::FlexFlow::nonempty_unordered_set<T>>(
        gen::nonEmpty<std::unordered_set<T>>());
  }
};

} // namespace rc

namespace std {

template <typename T>
struct hash<::FlexFlow::nonempty_unordered_set<T>> {
  size_t operator()(::FlexFlow::nonempty_unordered_set<T> const &x) const {
    return ::FlexFlow::get_std_hash(x.unwrap_as_unordered_set());
  };
};

} // namespace std

#endif
