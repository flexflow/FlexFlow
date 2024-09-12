#include "utils/join_strings.h"
#include "test/utils/doctest.h"
#include <string>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("join_strings") {
    std::vector<std::string> const v = {"Hello", "world", "!"};

    SUBCASE("iterator") {
      CHECK(join_strings(v.begin(), v.end(), " ") == "Hello world !");
    }

    SUBCASE("join_strings with container") {
      CHECK(join_strings(v, " ") == "Hello world !");
    }

    SUBCASE("join_strings with transforming function") {
      auto add_exclamation = [](std::string const &str) { return str + "!"; };
      CHECK(join_strings(v, " ", add_exclamation) == "Hello! world! !!");
    }

    SUBCASE("join_strings with transforming function, iterator") {
      auto add_exclamation = [](std::string const &str) { return str + "!"; };
      CHECK(join_strings(v.begin(), v.end(), " ", add_exclamation) ==
            "Hello! world! !!");
    }
  }
}
