#include "utils/bidict/algorithms/merge.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_maps(bidict<L, R> const &, bidict<L, R> const &)") {
    bidict<int, std::string> b1;
    b1.equate(1, "one");
    b1.equate(2, "two");

    bidict<int, std::string> b2;
    b2.equate(4, "four");

    bidict<int, std::string> correct;
    correct.equate(1, "one");
    correct.equate(2, "two");
    correct.equate(4, "four");

    CHECK(merge_maps(b1, b2) == correct);
  }
}
