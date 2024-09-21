#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_OPTIONAL_H

#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename T>
struct StringMaker<std::optional<T>> {
  static String convert(std::optional<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
