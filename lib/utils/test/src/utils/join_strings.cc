#include "utils/join_strings.h"
#include <doctest/doctest.h>
#include <string>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("join_strings") {
    std::vector<std::string> v = {"Hello", "world", "!"};

    SUBCASE("iterator") {
      std::string result = join_strings(v.begin(), v.end(), " ");
      std::string correct = "Hello world !";
      CHECK(result == correct);
    }

    SUBCASE("join_strings with container") {
      std::string result = join_strings(v, " ");
      std::string correct = "Hello world !";
      CHECK(result == correct);
    }

    SUBCASE("join_strings with transforming function") {
      auto add_exclamation = [](std::string const &str) { return str + "!"; };
      std::string result = join_strings(v, " ", add_exclamation);
      std::string correct = "Hello! world! !!";
      CHECK(result == correct);
    }

    SUBCASE("join_strings with transforming function, iterator") {
      auto add_exclamation = [](std::string const &str) { return str + "!"; };
      std::string result =
          join_strings(v.begin(), v.end(), " ", add_exclamation);
      std::string correct = "Hello! world! !!";
      CHECK(result == correct);
    }

    SUBCASE("empty sequence") {
      v = {};
      std::string result = join_strings(v, "!");
      std::string correct = "";
      CHECK(result == correct);
    }
  }
}
