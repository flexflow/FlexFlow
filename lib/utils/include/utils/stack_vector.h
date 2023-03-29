#ifndef _FLEXFLOW_UTILS_STACK_VECTOR_H
#define _FLEXFLOW_UTILS_STACK_VECTOR_H

#include <array>
#include <cassert>
#include "hash-utils.h"

namespace FlexFlow {

template <typename T, std::size_t MAXSIZE>
struct stack_vector {
public:
  stack_vector() = default;

  template <typename Iterator>
  stack_vector(Iterator start, Iterator end) {
    assert (end - start >= 0);
    assert (end - start <= MAXSIZE);
    for (; start < end; start++) {
      this->push_back(*start);
    }
  }

  void push_back(T const &t) {
    assert (this->m_size < MAXSIZE);
    this->contents[this->m_size] = t;
    this->m_size++;
  }

  template< class... Args >
  void emplace_back( Args&&... args ) {
    this->contents.emplace_back(std::forward<Args>(args)...);
  }

  T const &back() const {
    assert (this->m_size >= 1);
    return this->contents[this->m_size-1];
  }

  T &back() {
    assert (this->m_size >= 1);
    return this->contents[this->m_size-1];
  }

  T const &at(std::size_t idx) const {
    assert (idx < MAXSIZE);
    return this->contents[idx];
  }

  T &at(std::size_t idx) {
    assert (idx < MAXSIZE);
    return this->contents[idx];
  }

  T const &operator[](std::size_t idx) const {
    return this->at(idx);
  }

  T &operator[](std::size_t idx) {
    return this->at(idx);
  }

  using iterator = typename std::array<T, MAXSIZE>::iterator;
  using const_iterator = typename std::array<T, MAXSIZE>::const_iterator;
  using value_type = T;
  using reference = T&;
  using const_reference = T const &;

  iterator begin() {
    return this->contents.begin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cbegin() const {
    return this->contents.cbegin();
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

  std::size_t size() const {
    return this->m_size;
  }

  T *data() {
    return this->contents.data();
  }

  T const *data() const {
    return this->contents.data();
  }
private:
  std::size_t m_size = 0;
  std::array<T, MAXSIZE> contents;
};

}

namespace std {

using ::FlexFlow::stack_vector;

template <typename T, std::size_t MAXSIZE>
struct hash<stack_vector<T, MAXSIZE>> {
  size_t operator()(stack_vector<T, MAXSIZE> const &v) {
    size_t result = 0;
    hash_combine(result, v.size());
    for (auto const &ele : v) {
      hash_combine(result, ele);
    }
    return result;
  }
};

}

#endif
