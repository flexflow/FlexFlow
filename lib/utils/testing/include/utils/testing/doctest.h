#ifndef _FLEXFLOW_LIB_UTILS_TESTING_INCLUDE_UTILS_TESTING_DOCTEST_H
#define _FLEXFLOW_LIB_UTILS_TESTING_INCLUDE_UTILS_TESTING_DOCTEST_H

#include "doctest/doctest.h"
#include <type_traits>
#include "utils/preprocessor_extra/stringize.h"
#include <boost/type_index.hpp>

#define CHECK_TYPE_EQ(a, b) \
  CHECK_MESSAGE( \
    WRAP_ARG(std::is_same_v<UNWRAP_ARG(a), UNWRAP_ARG(b)>), \
    "Types do not match: ", \
    DOCTEST_TYPENAME(a), \
    " != ", \
    DOCTEST_TYPENAME(b));
#define CHECK_EXISTS(...) CHECK_SAME_TYPE(std::void_t<__VA_ARGS__>, void);
#define DOCTEST_TYPENAME(a) \
  ::FlexFlow::doctestGetTypeName<UNWRAP_ARG(a)>()

namespace FlexFlow {

template <typename T>
doctest::String doctestGetTypeName() {
  return doctest::String{boost::typeindex::type_id<T>().pretty_name().c_str()};
}

} // namespace FlexFlow

#endif
