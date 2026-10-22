#include "utils/nonempty_unordered_set/nonempty_unordered_set.h"
#include "test/utils/rapidcheck/doctest.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("nonempty_unordered_set") {
    SUBCASE("construct from initializer_list") {
      SUBCASE("does not throw if nonempty") {
        nonempty_unordered_set<int> s{1, 2, 3};

        CHECK(s.num_elements() == 3_p);
      }

      SUBCASE("throws if empty") {
        auto init_with_empty = []() -> void {
          nonempty_unordered_set<int> s{std::initializer_list<int>{}};
        };

        CHECK_THROWS(init_with_empty());
      }
    }

    SUBCASE("construct from unordered_set") {
      SUBCASE("does not throw if nonempty") {
        nonempty_unordered_set<int> s{
            std::unordered_set<int>{1, 2, 3},
        };

        CHECK(s.num_elements() == 3_p);
      }

      SUBCASE("throws if empty") {
        auto init_with_empty = []() -> void {
          nonempty_unordered_set<int> s{
              std::unordered_set<int>{},
          };
        };

        CHECK_THROWS(init_with_empty());
      }
    }
  }

  TEST_CASE("rc::Arbitrary<nonempty_unordered_set<T>>") {
    RC_SUBCASE([](nonempty_unordered_set<int>) {});
  }
}
