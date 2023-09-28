#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_INL
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_CONTAINERS_INL

#include "bidict.h"
#include "containers.decl.h"
#include "invoke.h"
#include "required_core.h"
#include "type_traits_core.h"
#include "utils/exception.h"
#include "utils/fmt.h"
#include "utils/optional.h"
#include "utils/sequence.h"
#include "utils/type_traits.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "utils/type_traits.h"

namespace FlexFlow {

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
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
std::unordered_set<std::pair<typename C::key_type, typename C::value_type>>
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
tl::optional<std::size_t> index_of(Container const &c, Element const &e) {
  auto it = std::find(c.cbegin(), c.cend(), e);
  if (it == c.cend()) {
    return tl::nullopt;
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
optional<T> intersection(C const &c) {
  optional<T> result;
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

template <typename F, typename C, typename K, typename V>
std::unordered_map<K, V> generate_map(C const &c, F const &f) {
  static_assert(is_hashable<K>::value,
                "Key type should be hashable (but is not)");

  auto transformed = transform(c, [&](K const &k) -> std::pair<K, V> {
    return {k, f(k)};
  });
  return {transformed.cbegin(), transformed.cend()};
}

template <typename K, typename V>
std::function<V(K const &)> lookup_in(std::unordered_map<K, V> const &m) {
  return [&m](K const &k) -> V { return m.at(k); };
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
bool is_supserseteq_of(std::unordered_set<T> const &l,
                       std::unordered_set<T> const &r) {
  return is_subseteq_of<T>(r, l);
}

template <typename C>
optional<typename C::value_type> maybe_get_only(C const &c) {
  if (c.size() == 1) {
    return *c.cbegin();
  } else {
    return nullopt;
  }
}

template <typename C>
typename C::value_type get_only(C const &c) {
  return unwrap(maybe_get_only(c), [&] {
    throw mk_runtime_error("Encountered container with size {} in get_only",
                           c.size());
  });
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
  std::vector<E> result(c.cbegin(), c.end());
  return result;
}

template <typename F, typename In, typename Out>
std::vector<Out> transform(std::vector<In> const &v, F const &f) {
  std::vector<Out> result;
  std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
  return result;
}

template <typename F, typename... Ts>
auto transform(std::tuple<Ts...> const &tup, F const &f) {
  return seq_transform(
      [&](auto idx) { return f(std::get<decltype(idx)::value>(tup)); },
      seq_enumerate_args_t<Ts...>{});
}

template <typename F, typename... Ts>
void for_each(std::tuple<Ts...> const &tup, F const &f) {
  seq_for_each([&](auto idx) { f(std::get<decltype(idx)::value>(tup)); },
               seq_enumerate_args_t<Ts...>{});
}

template <typename Head, typename... Rest>
enable_if_t<types_are_all_same_v<Head, Rest...>, std::vector<Head>>
    to_vector(std::tuple<Head, Rest...> const &tup) {
  std::vector<Head> result;
  for_each(tup, [&](Head const &h) { result.push_back(h); });
  return result;
}

template <typename F, typename C>
auto transform(req<C> const &c, F const &f)
    -> decltype(transform(std::declval<C>(), std::declval<F>())) {
  return transform(static_cast<C>(c), f);
}

template <typename F>
std::string transform(std::string const &s, F const &f) {
  std::string result;
  std::transform(s.cbegin(), s.cend(), std::back_inserter(result), f);
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

template <typename C, typename Enable>
struct get_element_type {
  using type = typename C::value_type;
};

template <typename T>
struct get_element_type<optional<T>> {
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


template <typename C, typename F>
C filter(C const &v, F const &f) {
  C result(v);
  inplace_filter(result, f);
  return result;
}

template <typename C, typename F, typename Elem>
void inplace_filter(C &v, F const &f) {
  std::remove_if(v.begin(), v.end(), [&](Elem const &e) { return !f(e); });
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
std::vector<T> value_all(std::vector<optional<T>> const &v) {
  return transform(v, [](optional<T> const &element) {
    return unwrap(element, [] {
      throw mk_runtime_error(
          "Encountered element without value in call to value_all");
    });
  });
}


} // namespace FlexFlow

#endif
