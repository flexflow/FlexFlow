#include "utils/testing.h"
#include "utils/bidict/algorithms/lookup.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("lookup_in_l(bidict<L, R> const &)") {
    bidict<int, std::string> b;
    b.equate(1, "one");
    CHECK(lookup_in_l(b)(1) == b.at_l(1));
  }

  TEST_CASE("lookup_in_r(bidict<L, R> const &)") {
    bidict<int, std::string> b;
    b.equate(1, "one");
    CHECK(lookup_in_r(b)("one") == b.at_r("one"));
  }
}
