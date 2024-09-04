#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_NO_DUPLICATES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_NO_DUPLICATES_H

#include <unordered_set>
#include "utils/exception.h"
#include <fmt/format.h>
#include <set>
#include "utils/fmt/unordered_multiset.h"
#include "utils/fmt/multiset.h"

namespace FlexFlow {

template <typename T>
std::unordered_set<T> require_no_duplicates(std::unordered_multiset<T> const &s) {
  std::unordered_set<T> result{s.cbegin(), s.cend()};

  if (result.size() != s.size()) {
    throw mk_runtime_error(fmt::format("require_no_duplicates encountered duplicate in set {}", s));
  }

  return result;
}

template <typename T>
std::set<T> require_no_duplicates(std::multiset<T> const &s) {
  std::set<T> result{s.cbegin(), s.cend()};

  if (result.size() != s.size()) {
    throw mk_runtime_error(fmt::format("require_no_duplicates encountered duplicate in set {}", s));
  }

  return result;
}

} // namespace FlexFlow

#endif
