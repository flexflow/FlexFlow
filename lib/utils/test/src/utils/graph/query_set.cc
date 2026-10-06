#include "utils/graph/query_set.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("includes") {
    SUBCASE("query is matchall") {
      query_set<int> query = query_set<int>::matchall();

      bool result = includes(query, 3);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("query is empty set") {
      query_set<int> query = query_set<int>::match_values_in(std::set<int>{});

      bool result = includes(query, 3);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("query is negated matchall") {
      query_set<int> query = query_set<int>::matchall().negated();

      bool result = includes(query, 3);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("query is negated empty set") {
      query_set<int> query =
          query_set<int>::match_values_in(std::set<int>{}).negated();

      bool result = includes(query, 3);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("query is nonempty set") {
      query_set<int> query = query_set<int>::match_values_in({0, 2, 3, 7, 10});

      SUBCASE("set contains element") {
        bool result = includes(query, 3);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("set does not contain element") {
        bool result = includes(query, 8);
        bool correct = false;

        CHECK(result == correct);
      }
    }

    SUBCASE("query is negated nonempty set") {
      query_set<int> query =
          query_set<int>::match_except_values_in({0, 2, 3, 7, 10});

      SUBCASE("set contains element") {
        bool result = includes(query, 3);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("set does not contain element") {
        bool result = includes(query, 8);
        bool correct = true;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("apply_query") {
    std::set<int> const domain = {1, 2, 3, 4, 6, 7, 8};

    SUBCASE("query is matchall") {
      query_set<int> query = query_set<int>::matchall();

      std::set<int> result = apply_query(query, domain);
      std::set<int> correct = domain;

      CHECK(result == correct);
    }

    SUBCASE("query is empty set") {
      query_set<int> query = query_set<int>::match_values_in(std::set<int>{});

      std::set<int> result = apply_query(query, domain);
      std::set<int> correct = std::set<int>{};

      CHECK(result == correct);
    }

    SUBCASE("query is negated matchall") {
      query_set<int> query = query_set<int>::matchall().negated();

      std::set<int> result = apply_query(query, domain);
      std::set<int> correct = std::set<int>{};

      CHECK(result == correct);
    }

    SUBCASE("query is negated empty set") {
      query_set<int> query =
          query_set<int>::match_values_in(std::set<int>{}).negated();

      std::set<int> result = apply_query(query, domain);
      std::set<int> correct = domain;

      CHECK(result == correct);
    }

    SUBCASE("query is nonempty set") {
      query_set<int> query = query_set<int>::match_values_in({0, 2, 3, 7, 10});

      std::set<int> result = apply_query(query, domain);
      std::set<int> correct = std::set<int>{2, 3, 7};

      CHECK(result == correct);
    }

    SUBCASE("query is negated nonempty set") {
      query_set<int> query =
          query_set<int>::match_except_values_in({0, 2, 3, 7, 10});

      std::set<int> result = apply_query(query, domain);
      std::set<int> correct = std::set<int>{1, 4, 6, 8};

      CHECK(result == correct);
    }
  }

  TEST_CASE("query_set") {
    SUBCASE("handles optional values correctly") {
      std::optional<int> nopt = std::nullopt;
      std::optional<int> three = 3;
      std::optional<int> five = 5;

      query_set<std::optional<int>> q1 =
          query_set<std::optional<int>>::matchall();

      query_set<std::optional<int>> q2 =
          query_set<std::optional<int>>::match_values_in(std::set{
              nopt,
              three,
          });

      query_set<std::optional<int>> q3 =
          query_set<std::optional<int>>::match_single_value(nopt);

      query_set<std::optional<int>> q4 =
          query_set<std::optional<int>>::match_none();

      CHECK(includes(q1, nopt));
      CHECK(includes(q1, three));
      CHECK(includes(q1, five));

      CHECK(includes(q2, nopt));
      CHECK(includes(q2, three));
      CHECK_FALSE(includes(q2, five));

      CHECK(includes(q3, nopt));
      CHECK_FALSE(includes(q3, three));
      CHECK_FALSE(includes(q3, five));

      CHECK_FALSE(includes(q4, nopt));
      CHECK_FALSE(includes(q4, three));
      CHECK_FALSE(includes(q4, five));
    }
  }
}
