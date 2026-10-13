#include "utils/bidict/unordered_bidict.h"
#include "test/utils/doctest/check_without_stringify.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict") {
    unordered_bidict<int, std::string> dict;
    dict.equate(1, "one");
    dict.equate(2, "two");

    SUBCASE("L type is the same as R type") {
      unordered_bidict<int, int> bd;
      bd.equate(1, 3);

      SUBCASE("unordered_bidict::contains_l") {
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("unordered_bidict::contains_r") {
        CHECK(bd.contains_r(3));
        CHECK_FALSE(bd.contains_r(1));
      }
    }

    SUBCASE("L type is not the same as R type") {
      unordered_bidict<int, std::string> bd;
      bd.equate(1, "one");
      bd.equate(2, "two");

      SUBCASE("unordered_bidict::contains_l") {
        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }

      SUBCASE("unordered_bidict::contains_r") {
        CHECK(bd.contains_r("one"));
        CHECK_FALSE(bd.contains_r("three"));
      }
    }

    SUBCASE("unordered_bidict::unordered_bidict(std::initializer_list<std::"
            "pair<L, R>>)") {
      unordered_bidict<int, std::string> bd{{1, "one"}, {2, "two"}};
      CHECK(bd.contains_l(1));
      CHECK_FALSE(bd.contains_l(3));

      SUBCASE("invalid mapping") {
        CHECK_THROWS(
            unordered_bidict<int, std::string>{{1, "one"}, {2, "one"}});
      }
    }

    SUBCASE("unordered_bidict::unordered_bidict(InputIt)") {
      std::vector<std::pair<int, std::string>> pairs = {{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd{pairs.begin(), pairs.end()};
      CHECK(bd.contains_l(1));
      CHECK_FALSE(bd.contains_l(3));

      SUBCASE("invalid mapping") {
        std::vector<std::pair<int, std::string>> bad_pairs = {{1, "one"},
                                                              {2, "one"}};
        CHECK_THROWS(unordered_bidict<int, std::string>{bad_pairs.begin(),
                                                        bad_pairs.end()});
      }
    }

    SUBCASE("unordered_bidict::unordered_bidict(std::unordered_map<L, R> "
            "const &, std::unordered_map<R, L> const &)") {
      std::unordered_map<int, std::string> fwd = {{1, "one"}, {2, "two"}};
      std::unordered_map<std::string, int> bwd = {{"one", 1}, {"two", 2}};
      unordered_bidict<int, std::string> bd{fwd, bwd};
      CHECK(bd.contains_l(1));
      CHECK_FALSE(bd.contains_l(3));

      SUBCASE("invalid mapping") {
        std::unordered_map<int, std::string> bad_fwd = {{1, "one"}, {2, "one"}};
        std::unordered_map<std::string, int> bad_bwd = {{"one", 1}};
        CHECK_THROWS(unordered_bidict<int, std::string>{bad_fwd, bad_bwd});
      }
    }

    SUBCASE("unordered_bidict::erase_l") {
      dict.erase_l(1);
      CHECK(dict.size() == 1);
      CHECK_THROWS(dict.at_l(1));
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("unordered_bidict::erase_r") {
      dict.erase_r("one");
      CHECK(dict.size() == 1);
      CHECK_THROWS(dict.at_r("one"));
      CHECK(dict.at_l(2) == "two");
    }

    SUBCASE("unordered_bidict::equate") {
      CHECK(dict.at_l(1) == "one");
      CHECK(dict.at_r("one") == 1);
      CHECK(dict.at_l(2) == "two");
      CHECK(dict.at_r("two") == 2);

      dict.equate(1, "three");
      CHECK(dict.at_l(1) == "three");
      CHECK(dict.at_r("three") == 1);
      CHECK_THROWS(dict.at_r("one"));
      CHECK(dict.at_l(2) == "two");
      CHECK(dict.at_r("two") == 2);

      dict.equate(3, "three");
      CHECK(dict.at_l(3) == "three");
      CHECK(dict.at_r("three") == 3);
      CHECK_THROWS(dict.at_l(1));
      CHECK(dict.at_l(2) == "two");
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("unordered_bidict::equate_strict") {
      CHECK_THROWS(dict.equate_strict(1, "three"));
      CHECK_THROWS(dict.equate_strict(3, "two"));

      dict.equate_strict(3, "three");
      CHECK(dict.at_l(3) == "three");
      CHECK(dict.at_r("three") == 3);
    }

    SUBCASE("unordered_bidict::operator==") {
      unordered_bidict<int, std::string> bd{{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd2{{1, "one"}, {3, "three"}};
      CHECK(dict == bd);
      CHECK_FALSE(dict == bd2);
    }

    SUBCASE("unordered_bidict::operator!=") {
      unordered_bidict<int, std::string> bd{{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd2{{1, "one"}, {3, "three"}};
      CHECK_FALSE(dict != bd);
      CHECK(dict != bd2);
    }

    SUBCASE("unordered_bidict::at_l") {
      CHECK(dict.at_l(1) == "one");
      CHECK_THROWS(dict.at_l(3));
    }

    SUBCASE("unordered_bidict::at_r") {
      CHECK(dict.at_r("one") == 1);
      CHECK_THROWS(dict.at_r("three"));
    }

    SUBCASE("unordered_bidict::left_values") {
      CHECK(dict.left_values() == std::unordered_set<int>{1, 2});
    }

    SUBCASE("unordered_bidict::right_values") {
      CHECK(dict.right_values() ==
            std::unordered_set<std::string>{"one", "two"});
    }

    SUBCASE("unordered_bidict::size") {
      CHECK(dict.size() == 2);
    }

    SUBCASE("unordered_bidict::empty") {
      CHECK_FALSE(dict.empty());
      unordered_bidict<int, std::string> empty{};
      CHECK(empty.empty());
    }

    SUBCASE("unordered_bidict::begin") {
      // note: std::unordered_map does not define iteration order, so we can
      // only check that begin() refers to one of the contained pairs
      auto it = dict.begin();
      CHECK(dict.contains(it->first, it->second));
    }

    SUBCASE("unordered_bidict::end") {
      auto it = dict.end();

      CHECK_WITHOUT_STRINGIFY(it == dict.end());
    }

    SUBCASE("unordered_bidict::reversed") {
      unordered_bidict<std::string, int> reversed_dict = dict.reversed();
      CHECK(reversed_dict.at_l("one") == 1);
      CHECK(reversed_dict.at_r(2) == "two");
    }

    SUBCASE("implicitly convert to std::unordered_map") {
      std::unordered_map<int, std::string> res = dict;
      std::unordered_map<int, std::string> expected = {{1, "one"}, {2, "two"}};
      CHECK(res == expected);
    }

    SUBCASE("fmt::to_string(unordered_bidict<int, std::string>)") {
      std::string result = fmt::to_string(dict);
      std::string correct = fmt::to_string(dict.as_unordered_map());
      CHECK(result == correct);
    }
  }

  TEST_CASE("adl_serializer<unordered_bidict<L, R>>") {
    unordered_bidict<int, std::string> deserialized =
        unordered_bidict<int, std::string>{
            {2, "hello"},
            {3, "goodbye"},
            {4, "yes"},
        };

    nlohmann::json serialized = std::vector<std::pair<int, std::string>>{
        {2, "hello"},
        {3, "goodbye"},
        {4, "yes"},
    };

    SUBCASE("to_json") {
      nlohmann::json result = deserialized;
      nlohmann::json correct = serialized;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      unordered_bidict<int, std::string> result = serialized;
      unordered_bidict<int, std::string> correct = deserialized;

      CHECK(result == correct);
    }
  }

  TEST_CASE("rc::Arbitrary<unordered_bidict<L, R>>") {
    RC_SUBCASE([](unordered_bidict<int, std::string>) {});
  }
}
