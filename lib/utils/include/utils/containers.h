#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_INL
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_INL

#include "bidict.h"
#include "containers.decl.h"
#include "required_core.h"
#include "type_traits_core.h"
#include "utils/containers/extend_vector.h"
#include "utils/containers/vector_transform.h"
#include "utils/exception.h"
#include "utils/type_traits.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename Container, typename Element>
Element sum(Container const &container) {
  Element result = 0;
  for (Element const &element : container) {
    result += element;
  }
  return result;
}

template <typename Container, typename ConditionF, typename Element>
Element sum_where(Container const &container, ConditionF const &condition) {
  Element result = 0;
  for (Element const &element : container) {
    if (condition(element)) {
      result += element;
    }
  }
  return result;
}

template <typename Container, typename Element>
Element product(Container const &container) {
  Element result = 1;
  for (Element const &element : container) {
    result *= element;
  }
  return result;
}

template <typename Container, typename ConditionF, typename Element>
Element product_where(Container const &container, ConditionF const &condition) {
  Element result = 1;
  for (Element const &element : container) {
    if (condition(element)) {
      result *= element;
    }
  }
  return result;
}

template <typename It>
typename It::value_type product(It begin, It end) {
  using Element = typename It::value_type;
  return std::accumulate(
      begin, end, 1, [](Element const &lhs, Element const &rhs) {
        return lhs * rhs;
      });
}

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
}

template <typename C>
bool contains_key(C const &m, typename C::key_type const &k) {
  return m.find(k) != m.end();
}

template <typename K, typename V>
bool contains_l(bidict<K, V> const &m, K const &k) {
  return m.contains_l(k);
}

template <typename K, typename V>
bool contains_r(bidict<K, V> const &m, V const &v) {
  return m.contains_r(v);
}

template <typename K, typename V, typename F, typename K2>
std::unordered_map<K2, V> map_keys(std::unordered_map<K, V> const &m,
                                   F const &f) {
  std::unordered_map<K2, V> result;
  for (auto const &kv : m) {
    result.insert({f(kv.first), kv.second});
  }
  return result;
}

template <typename K, typename V, typename F, typename K2>
bidict<K2, V> map_keys(bidict<K, V> const &m, F const &f) {
  bidict<K2, V> result;
  for (auto const &kv : m) {
    result.equate(f(kv.first), kv.second);
  }
  return result;
}

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter_keys(std::unordered_map<K, V> const &m,
                                     F const &f) {
  std::unordered_map<K, V> result;
  for (auto const &kv : m) {
    if (f(kv.first)) {
      result.insert(kv);
    }
  }
  return result;
}

template <typename K, typename V, typename F>
bidict<K, V> filter_values(bidict<K, V> const &m, F const &f) {
  std::unordered_map<K, V> result;
  for (auto const &kv : m) {
    if (f(kv.second)) {
      result.equate(kv);
    }
  }
  return result;
}

template <typename K, typename V, typename F, typename V2>
std::unordered_map<K, V2> map_values(std::unordered_map<K, V> const &m,
                                     F const &f) {
  std::unordered_map<K, V2> result;
  for (auto const &kv : m) {
    result.insert({kv.first, f(kv.second)});
  }
  return result;
}

template <typename K, typename V, typename F, typename V2>
bidict<K, V2> map_values(bidict<K, V> const &m, F const &f) {
  bidict<K, V2> result;
  for (auto const &kv : m) {
    result.equate({kv.first, f(kv.second)});
  }
  return result;
}

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter_values(std::unordered_map<K, V> const &m,
                                       F const &f) {
  std::unordered_map<K, V> result;
  for (auto const &kv : m) {
    if (f(kv.second)) {
      result.insert(kv);
    }
  }
  return result;
}

template <typename K, typename V>
bool is_submap(std::unordered_map<K, V> const &m,
               std::unordered_map<K, V> const &sub) {
  return restrict_keys(m, keys(sub)) == sub;
}

template <typename C>
std::unordered_set<typename C::key_type> keys(C const &c) {
  std::unordered_set<typename C::key_type> result;
  for (auto const &kv : c) {
    result.insert(kv.first);
  }
  return result;
}

template <typename C>
std::vector<typename C::mapped_type> values(C const &c) {
  std::vector<typename C::mapped_type> result;
  for (auto const &kv : c) {
    result.push_back(kv.second);
  }
  return result;
}

template <typename C>
std::unordered_set<std::pair<typename C::key_type, typename C::mapped_type>>
    items(C const &c) {
  return {c.begin(), c.end()};
}

template <typename C, typename T>
std::unordered_set<T> unique(C const &c) {
  return {c.cbegin(), c.cend()};
}

template <typename C, typename T>
std::unordered_set<T> without_order(C const &c) {
  return unique(c);
}

template <typename Container, typename Element>
std::optional<std::size_t> index_of(Container const &c, Element const &e) {
  auto it = std::find(c.cbegin(), c.cend(), e);
  if (it == c.cend()) {
    return std::nullopt;
  } else {
    return std::distance(c.cbegin(), it);
  }
}

template <typename T>
std::unordered_set<T> intersection(std::unordered_set<T> const &l,
                                   std::unordered_set<T> const &r) {
  std::unordered_set<T> result;
  for (T const &ll : l) {
    if (contains(r, ll)) {
      result.insert(ll);
    }
  }
  return result;
}

template <typename C, typename T>
std::optional<T> intersection(C const &c) {
  std::optional<T> result;
  for (T const &t : c) {
    result = intersection(result.value_or(t), t);
  }

  return result;
}

template <typename T>
bool are_disjoint(std::unordered_set<T> const &l,
                  std::unordered_set<T> const &r) {
  return intersection<T>(l, r).empty();
}

template <typename K, typename V>
std::unordered_map<K, V> restrict_keys(std::unordered_map<K, V> const &m,
                                       std::unordered_set<K> const &mask) {
  std::unordered_map<K, V> result;
  for (auto const &kv : m) {
    if (contains(mask, kv.first)) {
      result.insert(kv);
    }
  }
  return result;
}

template <typename K, typename V>
std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &lhs,
                                    std::unordered_map<K, V> const &rhs) {
  assert(are_disjoint(keys(lhs), keys(rhs)));

  std::unordered_map<K, V> result;
  for (auto const &kv : lhs) {
    result.insert(kv);
  }
  for (auto const &kv : rhs) {
    result.insert(kv);
  }

  return result;
}

template <typename K, typename V>
bidict<K, V> merge_maps(bidict<K, V> const &lhs, bidict<K, V> const &rhs) {
  assert(are_disjoint(keys(lhs), keys(rhs)));

  bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate(kv.first, kv.second);
  }

  return result;
}

template <typename C>
auto invert_map(C const &m) {
  std::unordered_map<typename C::mapped_type,
                     std::unordered_set<typename C::key_type>>
      m_inv;
  for (auto const &[key, value] : m) {
    m_inv[value].insert(key);
  }
  return m_inv;
}

template <typename F, typename C, typename K, typename V>
std::unordered_map<K, V> generate_map(C const &c, F const &f) {
  static_assert(is_hashable<K>::value,
                "Key type should be hashable (but is not)");

  auto transformed = transform(c, [&](K const &k) -> std::pair<K, V> {
    return {k, f(k)};
  });
  return {transformed.cbegin(), transformed.cend()};
}

template <typename F, typename C, typename K, typename V>
bidict<K, V> generate_bidict(C const &c, F const &f) {
  static_assert(is_hashable<K>::value,
                "Key type should be hashable (but is not)");
  static_assert(is_hashable<V>::value,
                "Value type should be hashable (but is not)");

  auto transformed = transform(c, [&](K const &k) -> std::pair<K, V> {
    return {k, f(k)};
  });
  return {transformed.cbegin(), transformed.cend()};
}

template <typename E>
std::optional<E> at_idx(std::vector<E> const &v, size_t idx) {
  if (idx >= v.size()) {
    return std::nullopt;
  } else {
    return v.at(idx);
  }
}

template <typename K, typename V>
std::function<V(K const &)> lookup_in(std::unordered_map<K, V> const &m) {
  return [&m](K const &k) -> V { return m.at(k); };
}

template <typename L, typename R>
std::function<R(L const &)> lookup_in_l(bidict<L, R> const &m) {
  return [&m](L const &l) -> R { return m.at_l(l); };
}

template <typename L, typename R>
std::function<L(R const &)> lookup_in_r(bidict<L, R> const &m) {
  return [&m](R const &r) -> L { return m.at_r(r); };
}

template <typename T>
std::unordered_set<T> set_union(std::unordered_set<T> const &l,
                                std::unordered_set<T> const &r) {
  std::unordered_set<T> result = l;
  result.insert(r.cbegin(), r.cend());
  return result;
}

template <typename T>
std::unordered_set<T> set_difference(std::unordered_set<T> const &l,
                                     std::unordered_set<T> const &r) {
  return filter(l, [&](T const &element) { return !contains(r, element); });
}

template <typename C, typename T>
std::unordered_set<T> set_union(C const &sets) {
  std::unordered_set<T> result;
  for (std::unordered_set<T> const &s : sets) {
    for (T const &element : s) {
      result.insert(element);
    }
  }
  return result;
}

template <typename T>
bool is_subseteq_of(std::unordered_set<T> const &l,
                    std::unordered_set<T> const &r) {
  if (l.size() > r.size()) {
    return false;
  }

  for (auto const &ll : l) {
    if (!contains(r, ll)) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool is_supserseteq_of(std::unordered_set<T> const &l,
                       std::unordered_set<T> const &r) {
  return is_subseteq_of<T>(r, l);
}

template <typename S, typename D>
std::unordered_set<D>
    map_over_unordered_set(std::function<D(S const &)> const &f,
                           std::unordered_set<S> const &input) {
  std::unordered_set<D> result;
  std::transform(
      input.cbegin(), input.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename C>
std::optional<typename C::value_type> maybe_get_only(C const &c) {
  if (c.size() == 1) {
    return *c.cbegin();
  } else {
    return std::nullopt;
  }
}

template <typename C>
typename C::value_type get_only(C const &c) {
  return unwrap(maybe_get_only(c), [&] {
    throw mk_runtime_error("Encountered container with size {} in get_only",
                           c.size());
  });
}

template <typename T>
T get_first(std::unordered_set<T> const &s) {
  return *s.cbegin();
}

template <typename T, typename C>
void extend(std::vector<T> &lhs, C const &rhs) {
  extend_vector(lhs, rhs);
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, typename C>
void extend(std::unordered_set<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename C, typename E>
void extend(C &lhs, std::optional<E> const &e) {
  if (e.has_value()) {
    return extend(lhs, e.value());
  }
}

template <typename C, typename F>
bool all_of(C const &c, F const &f) {
  for (auto const &v : c) {
    if (!f(v)) {
      return false;
    }
  }
  return true;
}

template <typename Container, typename Function>
std::optional<bool> optional_all_of(Container const &container,
                                    Function const &func) {
  for (auto const &element : container) {
    std::optional<bool> condition = func(element);
    if (!condition.has_value()) {
      return std::nullopt;
    }

    if (!condition.value()) {
      return false;
    }
  }
  return true;
}

template <typename C, typename F>
int count(C const &c, F const &f) {
  int result = 0;
  for (auto const &v : c) {
    if (f(v)) {
      result++;
    }
  }
  return result;
}

template <typename C>
bool are_all_same(C const &c) {
  auto const &first = *c.cbegin();
  for (auto const &v : c) {
    if (v != first) {
      return false;
    }
  }
  return true;
}

template <typename C, typename E>
std::vector<E> as_vector(C const &c) {
  std::vector<E> result(c.cbegin(), c.cend());
  return result;
}

template <typename C>
auto as_set(C const &c) {
  using E = typename C::value_type;
  std::set<E> result(c.cbegin(), c.cend());
  return result;
}

template <typename F, typename In, typename Out>
std::vector<Out> transform(std::vector<In> const &v, F const &f) {
  std::vector<Out> result;
  std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
  return result;
}

template <typename F, typename C>
auto transform(req<C> const &c, F const &f)
    -> decltype(transform(std::declval<C>(), std::declval<F>())) {
  return transform(static_cast<C>(c), f);
}

template <typename F, typename In, typename Out>
std::unordered_set<Out> transform(std::unordered_set<In> const &v, F const &f) {
  std::unordered_set<Out> result;
  for (auto const &e : v) {
    result.insert(f(e));
  }
  return result;
}

template <typename F>
std::string transform(std::string const &s, F const &f) {
  std::string result;
  std::transform(s.cbegin(), s.cend(), std::back_inserter(result), f);
  return result;
}

template <typename F, typename Out>
std::vector<Out> repeat(int n, F const &f) {
  assert(n >= 0);

  std::vector<Out> result;
  for (int i = 0; i < n; i++) {
    result.push_back(f());
  }
  return result;
}

template <typename T>
bidict<size_t, T> enumerate(std::unordered_set<T> const &c) {
  bidict<size_t, T> m;
  size_t idx = 0;
  for (auto const &v : c) {
    m.equate(idx++, v);
  }
  return m;
}

std::vector<size_t> count(size_t n);

template <typename In, typename F, typename Out>
std::vector<Out> flatmap(std::vector<In> const &v, F const &f) {
  std::vector<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename C>
auto pairs(C const &c) {
  using E = typename C::value_type;
  std::vector<std::pair<E, E>> pairs;
  for (auto it1 = c.begin(); it1 != c.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != c.end(); ++it2) {
      pairs.emplace_back(*it1, *it2);
    }
  }
  return pairs;
}

template <typename C, typename Enable>
struct get_element_type {
  using type = typename C::value_type;
};

template <typename T>
struct get_element_type<std::optional<T>> {
  using type = T;
};

template <typename T>
using get_element_type_t = typename get_element_type<T>::type;

template <typename In, typename F, typename Out>
std::unordered_set<Out> flatmap(std::unordered_set<In> const &v, F const &f) {
  std::unordered_set<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename Out, typename In>
std::unordered_set<Out> flatmap_v2(std::unordered_set<In> const &v,
                                   std::unordered_set<Out> (*f)(In const &)) {
  std::unordered_set<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename C>
void inplace_sorted(C &c) {
  CHECK_SUPPORTS_ITERATOR_TAG(std::random_access_iterator_tag, C);
  std::sort(c.begin(), c.end());
}

template <typename C>
auto sorted(C const &c) {
  using Elem = typename C::value_type;
  std::vector<Elem> result(c.begin(), c.end());
  inplace_sorted(result);
  return result;
}

template <typename C, typename F, typename Elem>
void inplace_sorted_by(C &c, F const &f) {
  CHECK_SUPPORTS_ITERATOR_TAG(std::random_access_iterator_tag, C);

  auto custom_comparator = [&](Elem const &lhs, Elem const &rhs) -> bool {
    return f(lhs, rhs);
  };
  std::sort(c.begin(), c.end(), custom_comparator);
}

template <typename C, typename F, typename Elem>
std::vector<Elem> sorted_by(C const &c, F const &f) {
  std::vector<Elem> result(c.begin(), c.end());
  inplace_sorted_by(result, f);
  return result;
}

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f) {
  return [=](T const &lhs, T const &rhs) { return f(lhs) < f(rhs); };
}

template <typename C, typename F>
C filter(C const &v, F const &f) {
  C result(v);
  inplace_filter(result, f);
  return result;
}

template <typename T, typename F>
std::unordered_set<T> filter(std::unordered_set<T> const &v, F const &f) {
  std::unordered_set<T> result;
  for (T const &t : v) {
    if (f(t)) {
      result.insert(t);
    }
  }
  return result;
}

template <typename C, typename F, typename Elem>
void inplace_filter(C &v, F const &f) {
  std::remove_if(v.begin(), v.end(), [&](Elem const &e) { return !f(e); });
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> vector_split(std::vector<T> const &v,
                                                       std::size_t idx) {
  assert(v.size() > idx);

  std::vector<T> prefix(v.begin(), v.begin() + idx);
  std::vector<T> postfix(v.begin() + idx, v.end());
  return {prefix, postfix};
}

template <typename C>
typename C::value_type maximum(C const &v) {
  return *std::max_element(v.begin(), v.end());
}

template <typename T>
T reversed(T const &t) {
  T r;
  for (auto i = t.cend() - 1; i >= t.begin(); i--) {
    r.push_back(*i);
  }
  return r;
}

template <typename T>
std::vector<T> value_all(std::vector<std::optional<T>> const &v) {
  return transform(v, [](std::optional<T> const &element) {
    return unwrap(element, [] {
      throw mk_runtime_error(
          "Encountered element without value in call to value_all");
    });
  });
}

template <typename T>
std::unordered_set<T> value_all(std::unordered_set<std::optional<T>> const &v) {
  return transform(v, [](std::optional<T> const &element) {
    return unwrap(element, [] {
      throw mk_runtime_error(
          "Encountered element without value in call to value_all");
    });
  });
}

template <typename T>
std::vector<T> subvec(std::vector<T> const &v,
                      std::optional<int> const &maybe_start,
                      std::optional<int> const &maybe_end) {
  auto begin_iter = v.cbegin();
  auto end_iter = v.cend();

  auto resolve_loc = [&](int idx) ->
      typename std::vector<T>::iterator::difference_type {
        if (idx < 0) {
          return v.size() + idx;
        } else {
          return idx;
        }
      };

  if (maybe_start.has_value()) {
    begin_iter += resolve_loc(maybe_start.value());
  }
  if (maybe_end.has_value()) {
    end_iter = v.cbegin() + resolve_loc(maybe_end.value());
  }

  std::vector<T> output(begin_iter, end_iter);
  return output;
}

template <typename C>
struct reversed_container_t {
  reversed_container_t() = delete;
  reversed_container_t(C const &c) : container(c) {}

  reversed_container_t(reversed_container_t const &) = delete;
  reversed_container_t(reversed_container_t &&) = delete;
  reversed_container_t &operator=(reversed_container_t const &) = delete;
  reversed_container_t &operator=(reversed_container_t &&) = delete;

  using iterator = typename C::reverse_iterator;
  using const_iterator = typename C::const_reverse_iterator;
  using reverse_iterator = typename C::iterator;
  using const_reverse_iterator = typename C::const_iterator;
  using value_type = typename C::value_type;
  using pointer = typename C::pointer;
  using const_pointer = typename C::const_pointer;
  using reference = typename C::reference;
  using const_reference = typename C::const_reference;

  iterator begin() {
    return this->container.rend();
  }

  iterator end() {
    return this->container.rbegin();
  }

  const_iterator cbegin() const {
    return this->container.crend();
  }

  const_iterator cend() const {
    return this->container.crbegin();
  }

  const_iterator begin() const {
    return this->cbegin();
  }

  const_iterator end() const {
    return this->cend();
  }

  reverse_iterator rbegin() {
    return this->container.begin();
  }

  reverse_iterator rend() {
    return this->container.end();
  }

  const_reverse_iterator crbegin() const {
    return this->container.cbegin();
  }

  const_reverse_iterator crend() const {
    return this->container.cend();
  }

  const_reverse_iterator rbegin() const {
    return this->crbegin();
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

private:
  C const &container;
};

template <typename C>
reversed_container_t<C> reversed_container(C const &c) {
  return reversed_container_t<C>(c);
}

} // namespace FlexFlow

#endif
