#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_BIDICT_H

#include "utils/fmt/unordered_map.h"
#include <cassert>
#include <unordered_map>
#include <optional>

namespace FlexFlow {

template <typename L, typename R>
struct bidict {
  bidict() : fwd_map{}, bwd_map{} {}

  bidict(std::initializer_list<std::pair<L, R>> init)
    : bidict(init.begin(), init.end())
  { }

  template <typename InputIt>
  bidict(InputIt first, InputIt last) {
    for (auto it = first; it != last; it++) {
      this->equate(it->first, it->second);
    }
  }

  bool contains_l(L const &l) const {
    return fwd_map.find(l) != fwd_map.end();
  }

  bool contains_r(R const &r) const {
    return bwd_map.find(r) != bwd_map.end();
  }

  void erase_l(L const &l) {
    this->fwd_map.erase(l);
    for (auto const &kv : this->bwd_map) {
      if (kv.second == l) {
        bwd_map.erase(kv.first);
        break;
      }
    }
  }

  void erase_r(R const &r) {
    this->bwd_map.erase(r);
    for (auto const &kv : this->fwd_map) {
      if (kv.second == r) {
        fwd_map.erase(kv.first);
        break;
      }
    }
  }

  void equate(L const &l, R const &r) {
    fwd_map.insert({l, r});
    bwd_map.insert({r, l});
  }

  void equate(std::pair<L, R> const &lr) {
    fwd_map.insert(lr);
    bwd_map.insert({lr.second, lr.first});
  }

  bool operator==(bidict<L, R> const &other) const {
    bool result = this->fwd_map == other.fwd_map;
    assert(result == (this->bwd_map == other.bwd_map));
    return result;
  }

  bool operator!=(bidict<L, R> const &other) const {
    bool result = this->fwd_map != other.fwd_map;
    assert(result == (this->bwd_map != other.bwd_map));
    return result;
  }

  R const &at_l(L const &l) const {
    return fwd_map.at(l);
  }

  L const &at_r(R const &r) const {
    return bwd_map.at(r);
  }

  std::size_t size() const {
    assert(fwd_map.size() == bwd_map.size());
    return fwd_map.size();
  }

  using const_iterator = typename std::unordered_map<L, R>::const_iterator;
  using value_type = std::pair<L, R>;
  using reference = value_type &;
  using const_reference = value_type const &;
  using key_type = L;
  using mapped_type = R;
  /* struct const_iterator { */
  /*   using iterator_category = std::forward_iterator_tag; */
  /*   using difference_type = std::size_t; */
  /*   using value_type = std::pair<L, R>; */
  /*   using pointer = std::pair<L, R> const *; */
  /*   using reference = std::pair<L, R> const &; */

  /*   explicit const_iterator(typename std::unordered_map<tl::optional<L>,
   * tl::optional<R>>::const_iterator); */

  /*   reference operator*() const { */
  /*     this->current = {this->it->first.value(), this->it->second.value()}; */
  /*     return this->current.value(); */
  /*   } */
  /*   pointer operator->() const { */
  /*     return &this->operator*(); */
  /*   } */

  /*   const_iterator& operator++() { */
  /*     ++this->it; */
  /*     return *this; */
  /*   } */
  /*   const_iterator operator++(int) { */
  /*     auto tmp = *this; */
  /*     ++(*this); */
  /*     return tmp; */
  /*   } */

  /*   bool operator==(const_iterator const &other) const { */
  /*     return this->it == other.it; */
  /*   } */
  /*   bool operator!=(const_iterator const &other) const { */
  /*     return this->it != other.it; */
  /*   } */
  /* private: */
  /*   mutable tl::optional<std::pair<L, R>> current; */
  /*   typename std::unordered_map<tl::optional<L>,
   * tl::optional<R>>::const_iterator it; */
  /* }; */

  /* const_iterator cbegin() const { */
  /*   return const_iterator(this->fwd_map.cbegin()); */
  /* } */

  /* const_iterator begin() const { */
  /*   return this->cbegin(); */
  /* } */

  /* const_iterator cend() const { */
  /*   return const_iterator(this->fwd_map.cend()); */
  /* } */

  /* const_iterator end() const { */
  /*   return this->cend(); */
  /* } */

  const_iterator cbegin() const {
    return this->fwd_map.cbegin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator cend() const {
    return this->fwd_map.cend();
  }

  const_iterator end() const {
    return this->cend();
  }

  bidict<R, L> reversed() const {
    return bidict<R, L>(bwd_map, fwd_map);
  }

  operator std::unordered_map<L, R> const &() const {
    return this->fwd_map;
  }

  std::unordered_map<L, R> const &as_unordered_map() const {
    return this->fwd_map;
  }

  bidict(std::unordered_map<L, R> const &fwd_map,
         std::unordered_map<R, L> const &bwd_map)
      : fwd_map(fwd_map), bwd_map(bwd_map) {}

private:
  friend struct bidict<R, L>;

  std::unordered_map<L, R> fwd_map;
  std::unordered_map<R, L> bwd_map;
};

template <typename L, typename R>
std::unordered_map<L, R> format_as(bidict<L, R> const &b) {
  return b;
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, bidict<L, R> const &b) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return s << fmt::to_string(b);
}

template <typename K, typename V, typename F, 
          typename K2 = decltype(std::declval<F>()(std::declval<K>()))>
bidict<K2, V> map_keys(bidict<K, V> const &m, F const &f) {
  bidict<K2, V> result;
  for (auto const &kv : m) {
    result.equate(f(kv.first), kv.second);
  }
  return result;
}

template <typename K, typename V, typename F,
          typename V2 = decltype(std::declval<F>()(std::declval<V>()))>
bidict<K, V2> map_values(bidict<K, V> const &m, F const &f) {
  bidict<K, V2> result;
  for (auto const &kv : m) {
    result.equate({kv.first, f(kv.second)});
  }
  return result;
}

template <typename K, typename V, typename F>
bidict<K, V> filter_keys(bidict<K, V> const &m, F const &f) {
  bidict<K, V> result;
  for (auto const &kv : m) {
    if (f(kv.first)) {
      result.equate(kv);
    }
  }
  return result;
}

template <typename K, typename V, typename F>
bidict<K, V> filter_values(bidict<K, V> const &m, F const &f) {
  bidict<K, V> result;
  for (auto const &kv : m) {
    if (f(kv.second)) {
      result.equate(kv);
    }
  }
  return result;
}

template <typename K, typename V, typename F,
          typename K2 = typename std::invoke_result_t<F, K>::value_type>
bidict<K2, V> filtermap_keys(bidict<K, V> const &m, F const &f) {
  bidict<K2, V> result;
  for (auto const &[k, v] : m) {
    std::optional<K2> new_k = f(k);
    if (new_k.has_value()) {
      result.equate(new_k.value(), v);
    }
  }
  return result;
}

template <typename K, typename V, typename F,
          typename V2 = typename std::invoke_result_t<F, V>::value_type>
bidict<K, V2> filtermap_values(bidict<K, V> const &m, F const &f) {
  bidict<K, V2> result;
  for (auto const &[k, v] : m) {
    std::optional<V2> new_v = f(v);
    if (new_v.has_value()) {
      result.equate(k, new_v.value());
    }
  }
  return result;
}

template <typename K, typename V, typename F,
          typename K2 = typename std::invoke_result_t<F, K, V>::first_type,
          typename V2 = typename std::invoke_result_t<F, K, V>::second_type>
bidict<K2, V2> transform(bidict<K, V> const &m, F const &f) {
  bidict<K2, V2> result;
  for (auto const &[k, v] : m) {
    result.equate(f(k, v));
  }
  return result;
}

} // namespace FlexFlow

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::bidict<L, R>> {
  size_t operator()(::FlexFlow::bidict<L, R> const &b) const {
    return hash<unordered_map<L, R>>{}(b);
  }
};

} // namespace std

#endif
