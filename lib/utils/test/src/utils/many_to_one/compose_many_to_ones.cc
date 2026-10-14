#include "utils/many_to_one/compose_many_to_ones.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compose_many_to_ones") {
    SUBCASE("correct usage") {
      ManyToOne<std::string, int> fst = ManyToOne<std::string, int>{
          {{"one", "ONE"}, 1},
          {{"six", "SIX"}, 6},
          {{"th"}, 3},
      };

      ManyToOne<int, char> snd = ManyToOne<int, char>{
          {{1, 3}, 'o'},
          {{6}, 'e'},
      };

      ManyToOne<std::string, char> result = compose_many_to_ones(fst, snd);

      ManyToOne<std::string, char> correct = ManyToOne<std::string, char>{
          {{"one", "ONE", "th"}, 'o'},
          {{"six", "SIX"}, 'e'},
      };

      CHECK(result == correct);
    }

    SUBCASE("domains don't match") {
      ManyToOne<std::string, int> fst = ManyToOne<std::string, int>{
          {{"one", "ONE"}, 1},
          {{"six", "SIX"}, 6},
          {{"th"}, 3},
      };

      ManyToOne<int, char> snd = ManyToOne<int, char>{
          {{1, 3}, 'o'},
          {{4, 6}, 'e'},
      };

      CHECK_THROWS(compose_many_to_ones(fst, snd));
    }
  }
}
