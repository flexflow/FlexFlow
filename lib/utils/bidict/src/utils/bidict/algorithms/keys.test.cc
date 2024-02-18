#include "utils/testing.h"
#include "utils/bidict/algorithms/keys.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("keys_l(bidict<L, R> const &)") {
    bidict<int, std::string> b;
    b.equate(1, "one");
    b.equate(2, "two");
    CHECK(keys_l(b) == std::unordered_set<int>{1, 2});
  }

  TEST_CASE("keys_r(bidict<L, R> const &)") {
    bidict<int, std::string> b;
    b.equate(1, "one");
    b.equate(2, "two");
    CHECK(keys_r(b) == std::unordered_set<std::string>{"one", "two"});
  }
}
