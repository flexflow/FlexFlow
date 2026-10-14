#include "utils/bidict/algorithms/compose_bidicts.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compose_bidicts") {
    SUBCASE("correct usage") {
      bidict<int, std::string> fst = bidict<int, std::string>{
          {1, "one"},
          {2, "two"},
      };

      bidict<std::string, std::pair<std::string, std::string>> snd =
          bidict<std::string, std::pair<std::string, std::string>>{
              {"one", {"one", "ONE"}},
              {"two", {"two", "TWO"}},
          };

      bidict<int, std::pair<std::string, std::string>> result =
          compose_bidicts(fst, snd);

      bidict<int, std::pair<std::string, std::string>> correct =
          bidict<int, std::pair<std::string, std::string>>{
              {1, {"one", "ONE"}},
              {2, {"two", "TWO"}},
          };

      CHECK(result == correct);
    }

    SUBCASE("domains don't match") {
      bidict<int, std::string> fst = bidict<int, std::string>{
          {1, "one"},
          {2, "two"},
          {4, "four"},
      };

      bidict<std::string, std::pair<std::string, std::string>> snd =
          bidict<std::string, std::pair<std::string, std::string>>{
              {"one", {"one", "ONE"}},
              {"two", {"two", "TWO"}},
              {"three", {"three", "THREE"}},
          };

      CHECK_THROWS(compose_bidicts(fst, snd));
    }
  }
}
