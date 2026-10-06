#include "utils/graph/biindex.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BiIndex") {
    BiIndex<int, int, std::string> b;

    b.add_value(1, 1, "11");
    b.add_value(1, 2, "12");
    b.add_value(1, 3, "13");
    b.add_value(2, 1, "21");
    b.add_value(2, 2, "22");
    b.add_value(2, 3, "23");
    b.add_value(3, 1, "31");
    b.add_value(3, 2, "32");
    b.add_value(3, 3, "33");
    b.add_value(4, 1, "41");
    // intentionally no 42
    b.add_value(4, 3, "43");

    SUBCASE("query for everything") {
      std::set<std::string> result =
          b.query(query_set<int>::matchall(), query_set<int>::matchall());

      std::set<std::string> correct = {
          "11", "12", "13", "21", "22", "23", "31", "32", "33", "41", "43"};

      CHECK(result == correct);
    }

    SUBCASE("query for k1") {
      std::set<std::string> result = b.query(
          query_set<int>::match_single_value(2), query_set<int>::matchall());

      std::set<std::string> correct = {"21", "22", "23"};

      CHECK(result == correct);
    }

    SUBCASE("query for k2") {
      std::set<std::string> result = b.query(
          query_set<int>::matchall(), query_set<int>::match_single_value(2));

      std::set<std::string> correct = {"12", "22", "32"};

      CHECK(result == correct);
    }

    SUBCASE("negated query for k1") {
      std::set<std::string> result =
          b.query(query_set<int>::match_except_single_value(2),
                  query_set<int>::matchall());

      std::set<std::string> correct = {
          "11", "12", "13", "31", "32", "33", "41", "43"};

      CHECK(result == correct);
    }

    SUBCASE("negated query for k2") {
      std::set<std::string> result =
          b.query(query_set<int>::matchall(),
                  query_set<int>::match_except_single_value(2));

      std::set<std::string> correct = {
          "11", "13", "21", "23", "31", "33", "41", "43"};

      CHECK(result == correct);
    }

    SUBCASE("query for k1 and k2") {
      std::set<std::string> result =
          b.query(query_set<int>::match_single_value(2),
                  query_set<int>::match_single_value(1));

      std::set<std::string> correct = {"21"};

      CHECK(result == correct);
    }

    SUBCASE("query for k1 and negated k2") {
      std::set<std::string> result =
          b.query(query_set<int>::match_single_value(2),
                  query_set<int>::match_except_single_value(1));

      std::set<std::string> correct = {"22", "23"};

      CHECK(result == correct);
    }

    SUBCASE("query for negated k1 and k2") {
      std::set<std::string> result =
          b.query(query_set<int>::match_except_single_value(2),
                  query_set<int>::match_single_value(1));

      std::set<std::string> correct = {"11", "31", "41"};

      CHECK(result == correct);
    }

    SUBCASE("query for negated k1 and negated k2") {
      std::set<std::string> result =
          b.query(query_set<int>::match_except_single_value(2),
                  query_set<int>::match_except_single_value(1));

      std::set<std::string> correct = {"12", "13", "32", "33", "43"};

      CHECK(result == correct);
    }

    SUBCASE("query for thing that don't exist") {
      std::set<std::string> result =
          b.query(query_set<int>::match_single_value(7),
                  query_set<int>::match_single_value(5));

      std::set<std::string> correct = {};

      CHECK(result == correct);
    }
  }
}
