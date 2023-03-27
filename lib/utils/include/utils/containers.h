#ifndef _FLEXFLOW_UTILS_CONTAINERS_H
#define _FLEXFLOW_UTILS_CONTAINERS_H

#include <type_traits>
#include <string>
#include <sstream>
#include <functional>
#include <iostream>
#include "tl/optional.hpp"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cassert>
#include "bidict.h"
#include "stack_map.h"

namespace FlexFlow {

template <typename InputIt, typename F>
std::string join_strings(InputIt first, InputIt last, std::string const &delimiter, F const &f) {
  std::ostringstream oss;
  bool first_iter = true;
  /* int i = 0; */
  for (; first != last; first++) {
    if (!first_iter) {
      oss << delimiter;
    }
    oss << *first;
    /* break; */
    first_iter = false;
    /* i++; */
  }
  return oss.str();
}

template <typename InputIt>
std::string join_strings(InputIt first, InputIt last, std::string const &delimiter) {
  using Ref = typename InputIt::reference;
  return join_strings<InputIt>(first, last, delimiter, [](Ref r){ return r; });
}

template <typename Container>
std::string join_strings(Container const &c, std::string const &delimiter) {
  return join_strings(c.cbegin(), c.cend(), delimiter);
}

template <typename Container>
typename Container::const_iterator find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename Container, typename Element = typename Container::value_type>
Element sum(Container const &container) {
  Element result = 0;
  for (Element const &element : container) {
    result += element;
  }
  return result;
}

template <typename Container, typename Element = typename Container::value_type>
Element product(Container const &container) {
  Element result = 1;
  for (Element const &element : container) {
    result *= element;
  }
  return result;
}

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
}

template <typename K, typename V>
bool contains_key(std::unordered_map<K, V> const &m, K const &k) {
  return m.find(k) != m.end();
}

template <typename K, typename V, size_t MAXSIZE>
bool contains_key(stack_map<K, V, MAXSIZE> const &m, K const &k) {
  return m.find(k) != m.end();
}

template <typename K, typename V>
bool contains_l(bidict<K, V> const &m, K const &k) {
  return m.find(k) != m.end();
}

template <typename K, typename V>
bool contains_r(bidict<K, V> const &m, V const &v) {
  return m.find(v) != m.end();
}

template <typename C>
std::vector<typename C::key_type> keys(C const &c) {
  std::vector<typename C::key_type> result;
  for (auto const &kv : c) {
    result.push_back(kv.first);
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
std::unordered_set<T> intersection(std::unordered_set<T> const &l, std::unordered_set<T> const &r) {
  std::unordered_set<T> result;
  for (T const &ll : l) {
    if (contains(r, ll)) {
      result.insert(ll);
    }
  }
  return result;
}

template <typename T>
bool are_disjoint(std::unordered_set<T> const &l, std::unordered_set<T> const &r) {
  return intersection<T>(l, r).empty();
}

template <typename K, typename V>
std::unordered_map<K, V> restrict_keys(std::unordered_map<K, V> const &m, std::unordered_set<K> const &mask) {
  std::unordered_map<K, V> result;
  for (auto const &kv : m) {
    if (contains(mask, kv.first)) {
      result.insert(kv);
    }
  }
  return result;
}

template <typename K, typename V> 
std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &lhs, std::unordered_map<K, V> const &rhs) {
  assert (are_disjoint(keys(lhs), keys(rhs)));

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
  assert (are_disjoint(keys(lhs), keys(rhs)));

  bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate(kv.first, kv.second);
  }

  return result;
}

template <typename T> 
std::unordered_set<T> set_union(std::unordered_set<T> const &l, std::unordered_set<T> const &r) {
  std::unordered_set<T> result = l;
  result.insert(r.cbegin(), r.cend());
  return result;
}

template <typename C, typename T = typename C::value_type::value_type>
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
bool is_subseteq_of(std::unordered_set<T> const &l, std::unordered_set<T> const &r) {
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
bool is_supserseteq_of(std::unordered_set<T> const &l, std::unordered_set<T> const &r) {
  return is_subseteq_of<T>(r, l);
}

template <typename S, typename D>
std::unordered_set<D> map_over_unordered_set(std::function<D(S const &)> const &f, std::unordered_set<S> const &input) {
  std::unordered_set<D> result;
  std::transform(input.cbegin(), input.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename C>
typename C::value_type get_only(C const &c) {
  assert (c.size() == 1);
  return *c.cbegin(); 
}

template <typename T>
T get_first(std::unordered_set<T> const &s) {
  return *s.cbegin();
}

template <typename T>
void extend(std::vector<T> &lhs, std::vector<T> const &rhs) {
  lhs.reserve(lhs.size() + distance(rhs.begin(), rhs.end()));
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
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

template <typename F, typename In, typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::vector<Out> vector_transform(F const &f, std::vector<In> const &v) {
  std::vector<Out> result;
  std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
  return result;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> vector_split(std::vector<T> const &v, std::size_t idx) {
  assert (v.size() > idx);

  std::vector<T> prefix(v.begin(), v.begin() + idx);
  std::vector<T> postfix(v.begin() + idx, v.end());
  return { prefix, postfix };
}


template <typename T>
T maximum(std::vector<T> const &v) {
  return std::max_element(v.begin(), v.end());
}

template <typename T, size_t MAXSIZE>
T maximum(stack_vector<T, MAXSIZE> const &v) {
  return std::max_element(v.begin(), v.end());
}

}

#endif
