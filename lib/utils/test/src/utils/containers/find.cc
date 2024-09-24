#include "utils/containers/find.h"
#include "test/utils/doctest.h"
#include <algorithm>
#include <set>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find") {

    SUBCASE("vector") {
      std::vector<int> v = {1, 2, 3, 3, 4, 5, 3};

      SUBCASE("element found") {
        CHECK_WITHOUT_STRINGIFY(find(v, 3) == std::find(v.begin(), v.end(), 3));
      }

      SUBCASE("element not found") {
        CHECK_WITHOUT_STRINGIFY(find(v, 6) == std::find(v.begin(), v.end(), 6));
      }

      SUBCASE("multiple occurrences of element") {
        CHECK_WITHOUT_STRINGIFY(find(v, 3) == std::find(v.begin(), v.end(), 3));
      }
    }

    SUBCASE("unordered_set") {
      std::unordered_set<int> s = {1, 2, 3, 4, 5};

      SUBCASE("element found") {
        CHECK_WITHOUT_STRINGIFY(find(s, 3) == std::find(s.begin(), s.end(), 3));
      }

      SUBCASE("element not found") {
        CHECK_WITHOUT_STRINGIFY(find(s, 6) == std::find(s.begin(), s.end(), 6));
      }
    }

    SUBCASE("set") {
      std::set<int> s = {1, 2, 3, 4, 5};

      SUBCASE("element found") {
        CHECK_WITHOUT_STRINGIFY(find(s, 3) == std::find(s.begin(), s.end(), 3));
      }

      SUBCASE("element not found") {
        CHECK_WITHOUT_STRINGIFY(find(s, 6) == std::find(s.begin(), s.end(), 6));
      }
    }
  }
}
