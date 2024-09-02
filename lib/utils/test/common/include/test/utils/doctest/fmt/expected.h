#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_EXPECTED_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_EXPECTED_H

#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename T, typename E>
struct StringMaker<tl::expected<T, E>> {
  static String convert(tl::expected<T, E> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest


#endif
