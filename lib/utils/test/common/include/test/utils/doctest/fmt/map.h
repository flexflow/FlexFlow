#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_MAP_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_MAP_H

#include "utils/fmt/map.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename K, typename V>
struct StringMaker<std::map<K, V>> {
  static String convert(std::map<K, V> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
