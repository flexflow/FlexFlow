#include <doctest/doctest.h>
#include "utils/commutative_pair.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/contains.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("commutative_pair") {
    commutative_pair<int> x = {1, 2};
    commutative_pair<int> y = {2, 1};
    commutative_pair<int> z = {1, 1};

    SUBCASE("max and min") {
      SUBCASE("max") {
        CHECK(x.max() == 2);
      }

      SUBCASE("min") {
        CHECK(x.min() == 1);
      }

      RC_SUBCASE("max >= min", [](commutative_pair<int> const &p) {
        return p.max() >= p.min();
      });
    }

    SUBCASE("==") {
      CHECK(x == x);
      CHECK(x == y);
      CHECK_FALSE(x == z);

      RC_SUBCASE("== is reflexive", [](commutative_pair<int> const &p) {
        return p == p;
      });
    }

    SUBCASE("!=") {
      CHECK_FALSE(x != x);
      CHECK_FALSE(x != y);
      CHECK(x != z);

      RC_SUBCASE("!= is anti-reflexive", [](commutative_pair<int> const &p) {
        return !(p != p);
      });
    }

    SUBCASE("<") {
      CHECK_FALSE(x < x);
      CHECK_FALSE(x < y);
      CHECK(z < x);
      CHECK_FALSE(x < z);

      RC_SUBCASE("< uses left entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} < commutative_pair<int>{i1+1, i2}; 
      });

      RC_SUBCASE("< uses right entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} < commutative_pair<int>{i1, i2+1}; 
      });

      RC_SUBCASE("< is antisymmetric", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
        RC_PRE(p1 < p2);
        return !(p2 < p1);
      });


      RC_SUBCASE("< is anti-reflexive", [](commutative_pair<int> const &p) {
        return !(p < p);
      });

      RC_SUBCASE("< is transitive", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2, commutative_pair<int> const &p3) {
        RC_PRE(p1 < p2 && p2 < p3);
        return p1 < p3;
      });
    }

    SUBCASE(">") {
      CHECK_FALSE(x > x);
      CHECK_FALSE(x > y);
      CHECK(x > z);

      RC_SUBCASE("> uses left entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} > commutative_pair<int>{i1-1, i2}; 
      });

      RC_SUBCASE("> uses right entry", [](int i1, int i2) {
        return commutative_pair<int>{i1, i2} > commutative_pair<int>{i1, i2-1}; 
      });

      RC_SUBCASE("> is antireflexive", [](commutative_pair<int> const &p) {
        return !(p > p);
      });

      RC_SUBCASE("> is antisymmetric", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
        RC_PRE(p1 > p2);
        return !(p2 > p1);
      });

      RC_SUBCASE("> is transitive", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2, commutative_pair<int> const &p3) {
        RC_PRE(p1 < p2 && p2 < p3);
        return p1 < p3;
      });

      RC_SUBCASE("< implies flipped >", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
        RC_PRE(p1 < p2);
        return p2 > p1;
      });
    }

    SUBCASE("<=") {
      RC_SUBCASE("<= is reflexive", [](commutative_pair<int> const &p) {
        return p <= p;
      });

      RC_SUBCASE("< implies <=", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
        RC_PRE(p1 < p2);
        return p1 <= p2;
      });
    }

    SUBCASE(">=") {
      RC_SUBCASE(">= is reflexive", [](commutative_pair<int> const &p) {
        return p >= p;
      });

      RC_SUBCASE("> implies >=", [](commutative_pair<int> const &p1, commutative_pair<int> const &p2) {
        RC_PRE(p1 > p2);
        return p1 >= p2;
      });
    }

    SUBCASE("std::hash") {
      CHECK(get_std_hash(x) == get_std_hash(x));
      CHECK(get_std_hash(x) != get_std_hash(z));
    }

    SUBCASE("fmt::to_string") {
      std::string result = fmt::to_string(x);
      std::unordered_set<std::string> correct_options = {"{2, 1}", "{1, 2}"};
      CHECK(contains(correct_options, result));
    }

    SUBCASE("operator<<") {
      std::ostringstream oss;
      oss << x;
      std::string result = oss.str();
      std::unordered_set<std::string> correct_options = {"{2, 1}", "{1, 2}"};
      CHECK(contains(correct_options, result));
    }
  }
}
