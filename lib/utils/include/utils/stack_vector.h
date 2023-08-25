#ifndef _FLEXFLOW_UTILS_STACK_VECTOR_H
#define _FLEXFLOW_UTILS_STACK_VECTOR_H

#include "containers.h"
#include "hash-utils.h"
#include "optional.h"
#include "utils/fmt.h"
#include "utils/type_traits.h"
#include <array>
#include <cassert>
#include <type_traits>

namespace FlexFlow {

template <typename T, std::size_t MAXSIZE>
struct stack_vector {
public:
  stack_vector() = default;

  template <typename Iterator>
  stack_vector(Iterator start, Iterator end) {
    assert(end - start >= 0);
    assert(end - start <= MAXSIZE);
    for (; start < end; start++) {
      this->push_back(static_cast<T>(*start));
    }
  }

  operator std::vector<T>() {
    return {this->begin(), this->end()};
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
    return this->contents[this->m_size - 1].value();
  }

  T &back() {
    assert(this->m_size >= 1);
    return this->contents[this->m_size - 1].value();
  }

  T const &at(std::size_t idx) const {
    assert(idx < MAXSIZE);
    return this->contents[idx].value();
  }

  T &at(std::size_t idx) {
    assert(idx < MAXSIZE);
    return this->contents[idx].value();
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

    typename std::conditional<IS_CONST, optional<T> const *, optional<T> *>::
        type ptr;

    Iterator(typename std::conditional<IS_CONST,
                                       optional<T> const *,
                                       optional<T> *>::type ptr)
        : ptr(ptr) {}

    template <bool WAS_CONST,
              typename = typename std::enable_if<IS_CONST || !WAS_CONST>::type>
    Iterator(Iterator<WAS_CONST> const &rhs) : ptr(rhs.ptr) {}

    reference operator*() const {
      return ptr->value();
    }
    pointer operator->() const {
      return &ptr->value();
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
      return this->ptr[diff].value();
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
    optional<T> *ptr = this->contents.data();
    return iterator(ptr);
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cbegin() const {
    optional<T> const *ptr = this->contents.data();
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

private:
  std::size_t m_size = 0;
  std::array<optional<T>, MAXSIZE> contents;

  static_assert(
      implies<is_equal_comparable<T>, is_equal_comparable<stack_vector>>::value,
      "");
  static_assert(
      implies<is_neq_comparable<T>, is_neq_comparable<stack_vector>>::value,
      "");
  static_assert(
      implies<is_lt_comparable<T>, is_lt_comparable<stack_vector>>::value, "");
};

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

namespace fmt {

template <typename T, int MAXSIZE>
struct formatter<::FlexFlow::stack_vector<T, MAXSIZE>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::stack_vector<T, MAXSIZE> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result =
        "[" +
        join_strings(m.cbegin(),
                     m.cend(),
                     ", ",
                     [](T const &t) { return fmt::to_string(t); }) +
        "]";
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

#endif
