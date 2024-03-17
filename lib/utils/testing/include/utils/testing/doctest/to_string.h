#ifndef _FLEXFLOW_LIB_UTILS_TESTING_INCLUDE_UTILS_TESTING_DOCTEST_TO_STRING_H
#define _FLEXFLOW_LIB_UTILS_TESTING_INCLUDE_UTILS_TESTING_DOCTEST_TO_STRING_H

#include "utils/fmt_extra/is_fmtable.h"
#include "utils/fmt_extra/instances/vector.h"
#include "doctest/doctest.h"

namespace doctest {

template <typename T>
doctest::String usingFmt(T const &t) {
  CHECK_FMTABLE(T);
  std::string raw = fmt::to_string(t);
  return String{raw.c_str()};
}

template<typename T> 
struct StringMaker<std::vector<T>> {
  static String convert(std::vector<T> const& value) {
    return usingFmt(value);
  }
};

} // namespace doctest

#endif
