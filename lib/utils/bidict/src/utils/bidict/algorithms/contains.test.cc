#include "utils/bidict/algorithms/contains.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("contains_l(bidict<K, V> const &, K const &)") {
    bidict<int, int> b;
    b.equate(2, 3);
    CHECK(contains_l(b, 2));
    CHECK(!contains_l(b, 3)); // in rhs
    CHECK(!contains_l(b, 4)); // not contained at all
  }

  TEST_CASE("contains_r(bidict<K, V> const &, V const &)") {
    bidict<int, int> b;
    b.equate(2, 3);
    CHECK(contains_r(b, 3));
    CHECK(!contains_r(b, 2)); // in lhs
    CHECK(!contains_r(b, 4)); // not contained at all
  }
}
