#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_UNORDERED_MULTISET_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_UNORDERED_MULTISET_H

#include "utils/fmt/unordered_multiset.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename T>
struct StringMaker<std::unordered_multiset<T>> {
  static String convert(std::unordered_multiset<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
