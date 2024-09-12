#include "utils/hash-utils.h"
#include "test/utils/doctest.h"
#include "utils/hash/map.h"
#include "utils/hash/set.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"
#include "utils/hash/unordered_set.h"
#include "utils/hash/vector.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("hash-utils") {
    SUBCASE("vector") {
      std::vector<int> vec1{1, 2, 3};
      std::vector<int> vec2{1, 2, 3, 4};

      size_t hash1 = get_std_hash(vec1);
      size_t hash2 = get_std_hash(vec2);

      CHECK(hash1 != hash2);

      vec1.push_back(4);
      hash1 = get_std_hash(vec1);
      CHECK(hash1 == hash2);
    }

    SUBCASE("map") {
      std::map<int, int> map1{{1, 2}};
      std::map<int, int> map2{{1, 2}, {3, 4}};

      size_t hash1 = get_std_hash(map1);
      size_t hash2 = get_std_hash(map2);

      CHECK(hash1 != hash2);

      map1.insert({3, 4});
      hash1 = get_std_hash(map1);
      CHECK(hash1 == hash2);
    }

    SUBCASE("unordered_map") {
      std::unordered_map<int, int> map1{{1, 2}};
      std::unordered_map<int, int> map2{{1, 2}, {3, 4}};

      size_t hash1 = get_std_hash(map1);
      size_t hash2 = get_std_hash(map2);

      CHECK(hash1 != hash2);

      map1.insert({3, 4});
      hash1 = get_std_hash(map1);
      CHECK(hash1 == hash2);
    }

    SUBCASE("set") {
      std::set<int> set1{1, 2, 3};
      std::set<int> set2{1, 2, 3, 4};

      size_t hash1 = get_std_hash(set1);
      size_t hash2 = get_std_hash(set2);

      CHECK(hash1 != hash2);

      set1.insert(4);
      hash1 = get_std_hash(set1);
      CHECK(hash1 == hash2);
    }

    SUBCASE("unordered_set") {
      std::unordered_set<int> set1{1, 2, 3};
      std::unordered_set<int> set2{1, 2, 3, 4};

      size_t hash1 = get_std_hash(set1);
      size_t hash2 = get_std_hash(set2);

      CHECK(hash1 != hash2);

      set1.insert(4);
      hash1 = get_std_hash(set1);
      CHECK(hash1 == hash2);
    }

    SUBCASE("tuple") {
      std::tuple<int, std::string, double> tuple1{1, "test", 3.14};
      std::tuple<int, std::string, double> tuple2{2, "test", 3.14};

      size_t hash1 = get_std_hash(tuple1);
      size_t hash2 = get_std_hash(tuple2);

      CHECK(hash1 != hash2);

      std::get<0>(tuple1) = 2;
      hash1 = get_std_hash(tuple1);
      CHECK(hash1 == hash2);
    }
  }
}
