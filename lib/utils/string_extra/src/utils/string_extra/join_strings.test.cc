#include "utils/string_extra/join_strings.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("join_strings(InputIt, InputIt, std::string const &, F const &)") {
    std::vector<std::string> const v = {"Hello", "world", "!"};

    std::string result = join_strings(
        v.begin(), v.end(), " ", [](std::string const &s) { return s + "."; });
    std::string correct = "Hello. world. !.";

    CHECK(result == correct);
  }

  TEST_CASE("join_strings(InputIt, InputIt, std::string const &)") {
    std::vector<std::string> const v = {"Hello", "world", "!"};
    CHECK(join_strings(v.begin(), v.end(), " ") == "Hello world !");
  }

  TEST_CASE("join_strings(Container const &, std::string const &)") {
    std::vector<std::string> const v = {"Hello", "world"};
    CHECK(join_strings(v, " ") == "Hello world");
  }

  TEST_CASE("join_strings(Container const &, std::string const &, F const &)") {
    std::vector<std::string> const v = {"Hello", "world"};
    CHECK(join_strings(v, " ", [](std::string const &s) { return s + "."; }) ==
          "Hello. world.");
  }
}
