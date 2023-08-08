#ifndef _FLEXFLOW_UTILS_STACK_MAP_H
#define _FLEXFLOW_UTILS_STACK_MAP_H

#include "optional.h"
#include "stack_vector.h"

namespace std {

template <typename T1, typename T2>
ostream &operator<<(ostream &os, pair<T1, T2> const &p) {
  os << "{" << p.first << ", " << p.second << "}";
  return os;
}

} // namespace std

namespace FlexFlow {

template <typename K, typename V, std::size_t MAXSIZE>
struct stack_map {
  stack_map() = default;

  V &operator[](K const &k) {
    optional<size_t> idx = get_idx(k);
    if (!idx.has_value()) {
      this->contents.push_back({k, {}});
      idx = this->contents.size() - 1;
    }
    return this->contents.at(idx.value()).second;
  }

  operator std::vector<std::pair<K, V>>() {
    return {this->contents.begin(), this->contents.end()};
  }

  void insert(K const &k, V const &v) {
    optional<size_t> idx = get_idx(k);
    if (!idx.has_value()) {
      this->contents.push_back({k, v});
    } else {
      this->contents.at(idx.value()).second = v;
    }
  }

  V &at(K const &k) {
    return this->contents.at(get_idx(k).value()).second;
  }

  V const &at(K const &k) const {
    return this->contents.at(get_idx(k).value()).second;
  }

  using iterator = typename stack_vector<std::pair<K, V>, MAXSIZE>::iterator;
  using const_iterator =
      typename stack_vector<std::pair<K, V>, MAXSIZE>::const_iterator;
  using value_type = std::pair<K, V>;
  using reference = value_type &;
  using const_reference = value_type const &;
  using key_type = K;
  using mapped_type = V;

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

private:
  optional<size_t> get_idx(K const &k) const {
    for (std::size_t idx = 0; idx < contents.size(); idx++) {
      if (contents.at(idx).first == k) {
        return idx;
      }
    }

    return nullopt;
  }

  stack_vector<std::pair<K, V>, MAXSIZE> contents;
};

} // namespace FlexFlow

#endif
