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

namespace FlexFlow {
namespace utils {

template <typename InputIt, typename Stringifiable>
std::string join_strings(InputIt first, InputIt last, std::string const &delimiter, std::function<Stringifiable(InputIt)> const &f) {
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
  return join_strings<InputIt, typename InputIt::reference>(first, last, delimiter, [](Ref r){ return r; });
}

template <typename Container, typename Element>
typename Container::const_iterator find(Container const &c, Element const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename Container, typename Element>
bool contains(Container const &c, Element const &e) {
  return find<Container, Element>(c, e) != c.cend();
}

template <typename K, typename V>
bool contains_key(std::unordered_map<K, V> const &m, K const &kv) {
  return m.find(kv) != m.end();
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
std::unordered_set<T> set_union(std::unordered_set<T> const &l, std::unordered_set<T> const &r) {
  std::unordered_set<T> result = l;
  result.insert(r.cbegin(), r.cend());
  return result;
}

template <typename S, typename D>
std::unordered_set<D> map_over_unordered_set(std::function<D(S const &)> const &f, std::unordered_set<S> const &input) {
  std::unordered_set<D> result;
  std::transform(input.cbegin(), input.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename T>
T get_only(std::unordered_set<T> const &s) {
  assert (s.size() == 1);
  return *s.cbegin(); 
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

template <typename F, typename In, typename Out = decltype(std::declval<F>()(std::declval<In>()))>
std::vector<Out> vector_transform(F const &f, std::vector<In> const &v) {
  std::vector<Out> result;
  std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
  return result;
}

}
}

#endif
