#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_MULTISET_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_MULTISET_H

#include "utils/fmt/multiset.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename T>
struct StringMaker<std::multiset<T>> {
  static String convert(std::multiset<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
