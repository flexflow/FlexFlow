#include "utils/bidict/bidict.h"
#include "test/utils/doctest.h"
#include "utils/fmt/vector.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict") {
    bidict<int, std::string> dict;
    dict.equate(1, "one");
    dict.equate(2, "two");

    SUBCASE("bidict::contains_l") {
      SUBCASE("L type is not the same as R type") {
        CHECK(dict.contains_l(1));
        CHECK_FALSE(dict.contains_l(3));
      }

      SUBCASE("L type is the same as R type") {
        bidict<int, int> bd;
        bd.equate(1, 3);

        CHECK(bd.contains_l(1));
        CHECK_FALSE(bd.contains_l(3));
      }
    }

    SUBCASE("bidict::contains_r") {
      SUBCASE("L type is not the same as R type") {
        CHECK(dict.contains_r(std::string("one")));
        CHECK_FALSE(dict.contains_r(std::string("three")));
      }

      SUBCASE("L type is the same as R type") {
        bidict<int, int> bd;
        bd.equate(1, 3);

        CHECK(bd.contains_r(3));
        CHECK_FALSE(bd.contains_r(1));
      }
    }

    SUBCASE("bidict::contains_r, bidict::contains_r - same type") {
      bidict<int, int> bd;
      bd.equate(1, 3);
      bd.equate(2, 4);

      CHECK(bd.contains_l(1));
      CHECK_FALSE(bd.contains_l(3));
      CHECK(bd.contains_r(3));
      CHECK_FALSE(bd.contains_r(1));
    }

    SUBCASE("bidict::equate") {
      CHECK(dict.at_l(1) == "one");
      CHECK(dict.at_r("one") == 1);
      CHECK(dict.at_l(2) == "two");
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("bidict::erase_l") {
      dict.erase_l(1);
      CHECK(dict.size() == 1);
      CHECK_THROWS_AS(dict.at_l(1), std::out_of_range);
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("bidict::erase_r") {
      dict.erase_r("one");
      CHECK(dict.size() == 1);
      CHECK_THROWS_AS(dict.at_r("one"), std::out_of_range);
      CHECK(dict.at_l(2) == "two");
    }

    SUBCASE("bidict::reversed") {
      bidict<std::string, int> reversed_dict = dict.reversed();
      CHECK(reversed_dict.at_l("one") == 1);
      CHECK(reversed_dict.at_r(2) == "two");
    }

    SUBCASE("bidict::size") {
      CHECK(dict.size() == 2);
    }

    SUBCASE("implicitly convert to std::unordered_map") {
      std::unordered_map<int, std::string> res = dict;
      std::unordered_map<int, std::string> expected = {{1, "one"}, {2, "two"}};
      CHECK(res == expected);
    }

    SUBCASE("bidict::begin") {
      auto it = dict.begin();
      CHECK(it->first == 2);
      CHECK(it->second == "two");
    }

    SUBCASE("bidict::end") {
      auto it = dict.end();

      CHECK_WITHOUT_STRINGIFY(it == dict.end());
    }

    SUBCASE("map_keys(bidict<K, V>, F)") {
      bidict<std::string, std::string> result = map_keys(dict, [](int k) {
        std::ostringstream oss;
        oss << k;
        return oss.str();
      });
      bidict<std::string, std::string> correct = {
          {"1", "one"},
          {"2", "two"},
      };
      CHECK(result == correct);
    }

    SUBCASE("map_values(bidict<K, V>, F)") {
      bidict<int, std::string> result =
          map_values(dict, [](std::string const &v) { return v + "a"; });
      bidict<int, std::string> correct = {
          {1, "onea"},
          {2, "twoa"},
      };
      CHECK(result == correct);
    }

    SUBCASE("filter_keys(bidict<K, V>, F") {
      bidict<int, std::string> result =
          filter_keys(dict, [](int k) { return k == 1; });
      bidict<int, std::string> correct = {
          {1, "one"},
      };
      CHECK(result == correct);
    }

    SUBCASE("filter_values(bidict<K, V>, F") {
      bidict<int, std::string> result =
          filter_values(dict, [](std::string const &v) { return v == "two"; });
      bidict<int, std::string> correct = {
          {2, "two"},
      };
      CHECK(result == correct);
    }

    SUBCASE("filtermap_keys(bidict<K, V>, F)") {
      bidict<std::string, std::string> result =
          filtermap_keys(dict, [](int k) -> std::optional<std::string> {
            if (k == 1) {
              return std::nullopt;
            } else {
              std::ostringstream oss;
              oss << (k + 1);
              return oss.str();
            }
          });
      bidict<std::string, std::string> correct = {
          {"3", "two"},
      };
      CHECK(result == correct);
    }

    SUBCASE("filtermap_values(bidict<K, V>, F)") {
      bidict<int, int> result = filtermap_values(
          dict, [](std::string const &v) -> std::optional<int> {
            if (v == "two") {
              return std::nullopt;
            } else {
              return v.size() + 1;
            }
          });
      bidict<int, int> correct = {
          {1, 4},
      };
      CHECK(result == correct);
    }

    SUBCASE("transform(bidict<K, V>, F)") {
      bidict<std::string, int> result =
          transform(dict, [](int k, std::string const &v) {
            return std::make_pair(v, k);
          });
      bidict<std::string, int> correct = {
          {"one", 1},
          {"two", 2},
      };
      CHECK(result == correct);
    }

    SUBCASE("fmt::to_string(bidict<int, std::string>)") {
      std::string result = fmt::to_string(dict);
      std::string correct = fmt::to_string(dict.as_unordered_map());
      CHECK(result == correct);
    }
  }
}
