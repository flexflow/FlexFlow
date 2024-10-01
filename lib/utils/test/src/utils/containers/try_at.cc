#include "utils/containers/try_at.h"
#include <doctest/doctest.h>
#include <string>
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("try_at(T, K)", T, std::unordered_map<int, std::string>, std::map<int, std::string>) {
    T m = {{1, "one"}, {2, "two"}};

    SUBCASE("map contains key") {
      std::optional<std::string> result = try_at(m, 1);
      std::optional<std::string> correct = "one";

      CHECK(result == correct);
    }

    SUBCASE("map does not contain key") {
      std::optional<std::string> result = try_at(m, 3);
      std::optional<std::string> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
