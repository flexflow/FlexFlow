#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_VECTOR_H

#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename T>
struct StringMaker<std::vector<T>> {
  static String convert(std::vector<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
