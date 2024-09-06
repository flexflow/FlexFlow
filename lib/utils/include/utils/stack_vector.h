#ifndef _FLEXFLOW_UTILS_STACK_VECTOR_H
#define _FLEXFLOW_UTILS_STACK_VECTOR_H

#include "utils/hash-utils.h"
#include "utils/join_strings.h"
#include "utils/test_types.h"
#include "utils/type_traits.h"
#include <array>
#include <cassert>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <rapidcheck.h>
#include <type_traits>

namespace FlexFlow {

template <typename T, std::size_t MAXSIZE>
struct stack_vector {
private:
  using element_type = conditional_t<std::is_default_constructible<T>::value,
                                     T,
                                     std::optional<T>>;

  static T const &get_value(T const &t) {
    return t;
  }
  static T const &get_value(std::optional<T> const &t) {
    return t.value();
  }
  static T &get_value(T &t) {
    return t;
  }
  static T &get_value(std::optional<T> &t) {
    return t.value();
  }

public:
  stack_vector() = default;

  template <typename Iterator>
  stack_vector(Iterator start, Iterator end) {
    size_t elements_added = 0;
    for (; start != end; start++) {
      elements_added++;
      assert(elements_added <= MAXSIZE);
      this->push_back(static_cast<T>(*start));
    }
  }

  stack_vector(std::initializer_list<T> const &l)
      : stack_vector(l.begin(), l.end()) {}

  operator std::vector<T>() const {
    return {this->cbegin(), this->cend()};
  }

  void push_back(T const &t) {
    assert(this->m_size < MAXSIZE);
    this->contents[this->m_size] = t;
    this->m_size++;
  }

  template <class... Args>
  void emplace_back(Args &&...args) {
    assert(this->m_size < MAXSIZE);
    this->contents[this->m_size] = {std::forward<Args>(args)...};
    this->m_size++;
  }

  T const &back() const {
    assert(this->m_size >= 1);
    return get_value(this->contents[this->m_size - 1]);
  }

  T &back() {
    assert(this->m_size >= 1);
    return get_value(this->contents[this->m_size - 1]);
  }

  T const &at(std::size_t idx) const {
    assert(idx < MAXSIZE);
    return get_value(this->contents[idx]);
  }

  T &at(std::size_t idx) {
    assert(idx < MAXSIZE);
    return get_value(this->contents[idx]);
  }

  T const &operator[](std::size_t idx) const {
    return this->at(idx);
  }

  T &operator[](std::size_t idx) {
    return this->at(idx);
  }

  template <bool IS_CONST>
  struct Iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = typename std::conditional<IS_CONST, T const &, T &>::type;
    using pointer = typename std::conditional<IS_CONST, T const *, T *>::type;

    typename std::conditional<IS_CONST, element_type const *, element_type *>::
        type ptr;

    Iterator(typename std::conditional<IS_CONST,
                                       element_type const *,
                                       element_type *>::type ptr)
        : ptr(ptr) {}

    template <bool WAS_CONST,
              typename = typename std::enable_if<IS_CONST || !WAS_CONST>::type>
    Iterator(Iterator<WAS_CONST> const &rhs) : ptr(rhs.ptr) {}

    reference operator*() const {
      return get_value(*ptr);
    }
    pointer operator->() const {
      return &get_value(*ptr);
    }

    Iterator &operator++() {
      ptr++;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    Iterator &operator--() {
      ptr--;
      return *this;
    }
    Iterator operator--(int) {
      Iterator tmp = *this;
      --(*this);
      return tmp;
    }

    bool operator==(Iterator const &other) const {
      return this->ptr == other.ptr;
    }

    bool operator!=(Iterator const &other) const {
      return this->ptr != other.ptr;
    }

    reference operator+=(difference_type diff) const {
      ptr += diff;
      return *this;
    }
    Iterator operator+(difference_type diff) const {
      return {ptr + diff};
    }

    friend Iterator operator+(difference_type diff, Iterator it) {
      return it + diff;
    }

    reference operator-=(difference_type diff) const {
      ptr -= diff;
      return *this;
    }
    Iterator operator-(difference_type diff) const {
      return {ptr - diff};
    }

    difference_type operator-(Iterator const &rhs) const {
      return this->ptr - rhs.ptr;
    }

    reference operator[](difference_type const &diff) const {
      return get_value(this->ptr[diff]);
    }

    bool operator<(Iterator const &rhs) const {
      return this->ptr < rhs.ptr;
    }

    bool operator>(Iterator const &rhs) const {
      return this->ptr > rhs.ptr;
    }

    bool operator<=(Iterator const &rhs) const {
      return this->ptr <= rhs.ptr;
    }

    bool operator>=(Iterator const &rhs) const {
      return this->ptr >= rhs.ptr;
    }
  };

  using iterator = Iterator<false>;
  using const_iterator = Iterator<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using value_type = T;
  using reference = T &;
  using const_reference = T const &;

  iterator begin() {
    element_type *ptr = this->contents.data();
    return iterator(ptr);
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cbegin() const {
    element_type const *ptr = this->contents.data();
    return const_iterator(ptr);
  }

  iterator end() {
    return this->begin() + this->m_size;
  }

  const_iterator end() const {
    return this->cend();
  }

  const_iterator cend() const {
    return this->cbegin() + this->m_size;
  }

  reverse_iterator rbegin() {
    return std::reverse_iterator<iterator>(this->end());
  }

  const_reverse_iterator rbegin() const {
    return this->crbegin();
  }

  const_reverse_iterator crbegin() const {
    return std::reverse_iterator<const_iterator>(this->cend());
  }

  reverse_iterator rend() {
    return std::reverse_iterator<iterator>(this->begin());
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

  const_reverse_iterator crend() const {
    return std::reverse_iterator<const_iterator>(this->cbegin());
  }

  bool operator==(stack_vector<T, MAXSIZE> const &other) const {
    if (this->m_size != other.m_size) {
      return false;
    }
    for (std::size_t i = 0; i < this->m_size; i++) {
      if (other.at(i) != this->at(i)) {
        return false;
      }
    }

    return true;
  }

  bool operator!=(stack_vector<T, MAXSIZE> const &other) const {
    return !(*this == other);
  }

  bool operator<(stack_vector const &other) const {
    for (std::size_t i = 0; i < std::min(this->m_size, other.m_size); i++) {
      if (this->at(i) < other.at(i)) {
        return true;
      } else if (this->at(i) > other.at(i)) {
        return false;
      }
    }

    return (this->m_size < other.m_size);
  }

  std::size_t size() const {
    return this->m_size;
  }

  bool empty() const {
    return (this->m_size == 0);
  }

  T const *data() const {
    return this->contents.data();
  }

  friend std::string format_as(stack_vector<T, MAXSIZE> const &v) {
    CHECK_FMTABLE(T);

    std::string result =
        ::FlexFlow::join_strings(v.cbegin(), v.cend(), ", ", [](T const &t) {
          return fmt::to_string(t);
        });
    return "[" + result + "]";
  }

private:
  std::size_t m_size = 0;
  std::array<element_type, MAXSIZE> contents;

  static_assert(
      implies<is_equal_comparable<T>, is_equal_comparable<stack_vector>>::value,
      "");
  static_assert(
      implies<is_neq_comparable<T>, is_neq_comparable<stack_vector>>::value,
      "");
  static_assert(
      implies<is_lt_comparable<T>, is_lt_comparable<stack_vector>>::value, "");
};

template <typename T, std::size_t MAXSIZE>
std::ostream &operator<<(std::ostream &s, stack_vector<T, MAXSIZE> const &v) {
  return s << fmt::to_string(v);
}

template <typename T, std::size_t MAXSIZE>
void to_json(nlohmann::json &j, stack_vector<T, MAXSIZE> const &v) {
  std::vector<T> as_vec(v.begin(), v.end());
  j = as_vec;
}

template <typename T, std::size_t MAXSIZE>
void from_json(nlohmann::json const &j, stack_vector<T, MAXSIZE> &v) {
  std::vector<T> as_vec;
  j.get_to(as_vec);
  v = stack_vector<T, MAXSIZE>{as_vec.begin(), as_vec.end()};
}

} // namespace FlexFlow

namespace std {

template <typename T, std::size_t MAXSIZE>
struct hash<::FlexFlow::stack_vector<T, MAXSIZE>> {
  size_t operator()(::FlexFlow::stack_vector<T, MAXSIZE> const &v) {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "stack_vector elements must be hashable");
    size_t result = 0;
    iter_hash(result, v.cbegin(), v.cend());
    return result;
  }
};

} // namespace std

namespace rc {

template <typename T, std::size_t MAXSIZE>
struct Arbitrary<::FlexFlow::stack_vector<T, MAXSIZE>> {
  static Gen<::FlexFlow::stack_vector<T, MAXSIZE>> arbitrary() {
    return gen::mapcat(gen::inRange<size_t>(0, MAXSIZE), [](size_t size) {
      return gen::container<::FlexFlow::stack_vector<T, MAXSIZE>>(
          size, gen::arbitrary<T>());
    });
  }
};

} // namespace rc

#endif
