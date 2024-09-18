#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_PAIR_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_PAIR_H

#include "utils/fmt/pair.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename L, typename R>
struct StringMaker<std::pair<L, R>> {
  static String convert(std::pair<L, R> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
