#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_UNORDERED_MAP_H

#include "utils/fmt/unordered_map.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename K, typename V>
struct StringMaker<std::unordered_map<K, V>> {
  static String convert(std::unordered_map<K, V> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
