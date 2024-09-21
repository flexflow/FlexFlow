#include "utils/containers/to_uppercase.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("to_uppercase(std::string)") {
    std::string input = "Hello World";

    std::string result = to_uppercase(input);
    std::string correct = "HELLO WORLD";

    CHECK(result == correct);
  }
}
