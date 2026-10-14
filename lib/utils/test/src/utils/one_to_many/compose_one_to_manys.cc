#include "utils/one_to_many/compose_one_to_manys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compose_one_to_manys") {
    SUBCASE("correct usage") {
      OneToMany<int, std::string> fst = OneToMany<int, std::string>{
          {1, {"one", "ONE"}},
          {6, {"six", "SIX"}},
          {3, {"three"}},
      };

      OneToMany<std::string, char> snd = OneToMany<std::string, char>{
          {"one", {'o', 'n', 'e'}},
          {"ONE", {'O', 'N', 'E'}},
          {"six", {'s', 'i', 'x'}},
          {"SIX", {'S', 'I', 'X'}},
          {"three", {'t', 'h'}},
      };

      OneToMany<int, char> result = compose_one_to_manys(fst, snd);

      OneToMany<int, char> correct = OneToMany<int, char>{
          {1, {'o', 'n', 'e', 'O', 'N', 'E'}},
          {6, {'s', 'i', 'x', 'S', 'I', 'X'}},
          {3, {'t', 'h'}},
      };

      CHECK(result == correct);
    }

    SUBCASE("domains don't match") {
      OneToMany<int, std::string> fst = OneToMany<int, std::string>{
          {1, {"one", "ONE"}},
          {6, {"six", "SIX"}},
          {3, {"th"}},
      };

      OneToMany<std::string, char> snd = OneToMany<std::string, char>{
          {"one", {'o', 'n', 'e'}},
          {"ONE", {'O', 'N', 'E'}},
          {"six", {'s', 'i', 'x'}},
          {"three", {'t', 'h'}},
      };

      CHECK_THROWS(compose_one_to_manys(fst, snd));
    }
  }
}
