#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("merge_disjoint_bidicts") {

    SUBCASE("disjoint keys and values") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{3, "three"}, {4, "four"}};

      bidict<int, std::string> result = merge_disjoint_bidicts(bd1, bd2);
      bidict<int, std::string> correct = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};

      CHECK(result == correct);
    }

    SUBCASE("overlapping key, different associated value") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{2, "three"}, {3, "four"}};

      CHECK_THROWS(merge_disjoint_bidicts(bd1, bd2));
    }

    SUBCASE("overlapping key, same associated value") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{2, "two"}, {3, "three"}};

      CHECK_THROWS(merge_disjoint_bidicts(bd1, bd2));
    }

    SUBCASE("overlapping values") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{3, "two"}, {4, "four"}};

      CHECK_THROWS(merge_disjoint_bidicts(bd1, bd2));
    }
  }
}
