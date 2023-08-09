#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_FF_STACK_VECTOR_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_FF_STACK_VECTOR_H

#include "op-attrs/ff_dim.h"
#include "utils/stack_vector.h"

namespace FlexFlow {

template <typename Idx, typename T>
struct DimOrdered {
  DimOrdered() = delete;

  DimOrdered(std::initializer_list<T> const &l)
      : contents(l.begin(), l.end()) {}

  /* template <typename I, typename std::enable_if<std::is_convertible<I,
   * T>::value>::type> */
  DimOrdered(std::vector<T> const &contents)
      : contents(contents.begin(), contents.end()) {}

  /* template <typename It, typename std::enable_if<std::is_convertible<typename
   * It::value_type, T>::value>::type> */
  template <typename It>
  DimOrdered(It begin, It end) : contents(begin, end) {}

  template <size_t MAXSIZE>
  DimOrdered(stack_vector<T, MAXSIZE> const &contents)
      : contents(contents.begin(), contents.end()) {}

  T const &at(Idx idx) const {
    return this->contents.at(idx.value());
  }

  T &at(Idx idx) {
    return this->contents.at(idx.value());
  }

  T const &operator[](Idx idx) const {
    return this->at(idx);
  }

  T &operator[](Idx idx) {
    return this->at(idx);
  }

  bool operator==(DimOrdered const &other) const {
    return this->contents == other.contents;
  }

  bool operator!=(DimOrdered const &other) const {
    return this->contents != other.contents;
  }

  bool operator<(DimOrdered const &other) const {
    return this->contents < other.contents;
  }

  using iterator = typename stack_vector<T, MAX_TENSOR_DIM>::iterator;
  using const_iterator =
      typename stack_vector<T, MAX_TENSOR_DIM>::const_iterator;
  using reverse_iterator =
      typename stack_vector<T, MAX_TENSOR_DIM>::reverse_iterator;
  using const_reverse_iterator =
      typename stack_vector<T, MAX_TENSOR_DIM>::const_reverse_iterator;
  using value_type = T;
  using pointer = value_type *;
  using const_pointer = value_type const *;
  using reference = value_type &;
  using const_reference = value_type const &;

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
    return this->contents.end();
  }

  const_iterator end() const {
    return this->cend();
  }

  const_iterator cend() const {
    return this->contents.cend();
  }

  reverse_iterator rbegin() {
    return this->contents.rbegin();
  }

  const_reverse_iterator rbegin() const {
    return this->crbegin();
  }

  const_reverse_iterator crbegin() const {
    return this->contents.crbegin();
  }

  reverse_iterator rend() {
    return this->contents.crend();
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

  const_reverse_iterator crend() const {
    return this->contents.crend();
  }

  size_t size() const {
    return this->contents.size();
  }

  size_t num_dims() const {
    return this->size();
  }

  friend struct ::std::hash<DimOrdered>;

private:
  stack_vector<T, MAX_TENSOR_DIM> contents;
};

template <typename T>
using FFOrdered = DimOrdered<ff_dim_t, T>;

template <typename T>
auto inner_to_outer(FFOrdered<T> const &ff_ordered)
    -> decltype(reversed_container(ff_ordered)) {
  return reversed_container(ff_ordered);
}

template <typename T>
std::vector<ff_dim_t> inner_to_outer_idxs(FFOrdered<T> const &ff_ordered) {
  std::vector<ff_dim_t> idxs;
  for (size_t i = 0; i < ff_ordered.size(); i++) {
    idxs.push_back(ff_dim_t(ff_ordered.size() - i - 1));
  }
  return idxs;
}

template <typename T>
std::vector<ff_dim_t> outer_to_inner_idxs(FFOrdered<T> const &ff_ordered) {
  return reversed(inner_to_outer_idxs<T>(ff_ordered));
}

template <typename T>
FFOrdered<T> const &outer_to_inner(FFOrdered<T> const &ff_ordered) {
  return ff_ordered;
}

} // namespace FlexFlow

namespace std {

template <typename Idx, typename T>
struct hash<::FlexFlow::DimOrdered<Idx, T>> {
  size_t operator()(::FlexFlow::DimOrdered<Idx, T> const &t) const {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "Elements must be hashable");

    return get_std_hash(t.contents);
  }
};

} // namespace std

#endif
