#ifndef _FLEXFLOW_UTILS_STACK_MAP_H
#define _FLEXFLOW_UTILS_STACK_MAP_H

#include "utils/stack_vector.h"

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
    std::optional<size_t> idx = get_idx(k);
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
    std::optional<size_t> idx = get_idx(k);
    if (!idx.has_value()) {
      this->contents.push_back({k, v});
    } else {
      this->contents.at(idx.value()).second = v;
    }
  }

  size_t size() const {
    return this->contents.size();
  }

  bool empty() const {
    return this->contents.empty();
  }

  friend bool operator==(stack_map const &lhs, stack_map const &rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    return lhs.sorted() == rhs.sorted();
  }

  friend bool operator!=(stack_map const &lhs, stack_map const &rhs) {
    if (lhs.size() != rhs.size()) {
      return true;
    }
    return lhs.sorted() != rhs.sorted();
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
  std::vector<std::pair<K, V>> sorted() const {
    auto comparator = [](std::pair<K, V> const &lhs,
                         std::pair<K, V> const &rhs) {
      return lhs.first < rhs.first;
    };

    return sorted_by(this->contents, comparator);
  }

  std::optional<size_t> get_idx(K const &k) const {
    for (std::size_t idx = 0; idx < contents.size(); idx++) {
      if (contents.at(idx).first == k) {
        return idx;
      }
    }

    return std::nullopt;
  }

  stack_vector<std::pair<K, V>, MAXSIZE> contents;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(stack_map<int, int, 10>);

} // namespace FlexFlow

namespace doctest {

template <typename K, typename V, std::size_t MAXSIZE>
struct StringMaker<FlexFlow::stack_map<K, V, MAXSIZE>> {
  static String convert(FlexFlow::stack_map<K, V, MAXSIZE> const &map) {
    return toString(fmt::to_string(map));
  }
};

} // namespace doctest

#endif
