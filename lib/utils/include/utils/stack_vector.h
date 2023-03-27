#ifndef _FLEXFLOW_UTILS_STACK_VECTOR_H
#define _FLEXFLOW_UTILS_STACK_VECTOR_H

#include <array>
#include <cassert>

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
    assert (this->size < MAXSIZE);
    this->contents[this->size] = t;
    this->size++;
  }

  template< class... Args >
  void emplace_back( Args&&... args ) {
    this->contents.emplace_back(std::forward<Args>(args)...);
  }

  T const &back() const {
    assert (this->size >= 1);
    return this->contents[this->size-1];
  }

  T &back() {
    assert (this->size >= 1);
    return this->contents[this->size-1];
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
    return this->begin() + this->size;
  }

  const_iterator end() const {
    return this->cend();
  }

  const_iterator cend() const {
    return this->cbegin() + this->size;
  }
private:
  std::size_t size = 0;
  std::array<T, MAXSIZE> contents;
};

}

#endif
