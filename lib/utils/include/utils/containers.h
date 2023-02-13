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
bool contains(Container const &c, Element const &e) {
  return c.find(e) != c.cend();
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

template <typename S, typename D>
std::unordered_set<D> map_over_unordered_set(std::function<D(S const &)> const &f, std::unordered_set<S> const &input) {
  std::unordered_set<D> result;
  std::transform(input.cbegin(), input.cend(), std::inserter(result, result.begin()), f);
  return result;
}

#endif
