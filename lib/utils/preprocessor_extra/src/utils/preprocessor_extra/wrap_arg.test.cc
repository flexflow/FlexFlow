#include "utils/preprocessor_extra/wrap_arg.h"
#include "utils/testing.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#define EXAMPLE_MACRO(ARG1, ARG2, ...) BOOST_PP_STRINGIZE(UNWRAP_ARG(ARG1))

#define OTHER_EXAMPLE_MACRO(ARG1, ARG2, ...)                                   \
  BOOST_PP_STRINGIZE(UNWRAP_ARG(ARG2))

TEST_CASE("WRAP_ARG") {
  CHECK(EXAMPLE_MACRO(std::tuple<A, B, C>, H) == "std::tuple<A");
  CHECK(EXAMPLE_MACRO(WRAP_ARG(std::tuple<A, B, C>), H) ==
        "std::tuple<A, B, C>");
  CHECK(OTHER_EXAMPLE_MACRO(std::tuple<A, B, C>, std::tuple<D, E, F>) == "B");
  CHECK(OTHER_EXAMPLE_MACRO(WRAP_ARG(std::tuple<A, B, C>),
                            WRAP_ARG(std::tuple<D, E, F>)) ==
        "std::tuple<D, E, F>");
}

TEST_CASE("UNWRAP_ARG") {
  CHECK(BOOST_PP_STRINGIZE(UNWRAP_ARG(notwrapped)) == "notwrapped");
}
