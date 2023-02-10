#ifndef _FLEXFLOW_UTILS_CONTAINERS_H
#define _FLEXFLOW_UTILS_CONTAINERS_H

#include <type_traits>
#include <string>
#include <sstream>
#include <functional>
#include <iostream>

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
  return c.find(e) != c.end();
}

#endif
