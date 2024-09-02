#include <doctest/doctest.h>
#include "utils/disjoint_set.h"

using namespace FlexFlow;

template <typename T>
T generate_element(int seed);

template <>
int generate_element<int>(int seed) {
  return seed;
}

template <>
std::string generate_element<std::string>(int seed) {
  return "Element" + std::to_string(seed);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("DisjointSetUnionAndFind", T, int, std::string) {
    disjoint_set<optional<T>> ds;

    SUBCASE("SingleElementSets") {
      optional<T> element = generate_element<T>(1);
      CHECK(ds.find(element) == element);

      element = generate_element<T>(2);
      CHECK(ds.find(element) == element);
    }

    SUBCASE("UnionAndFind") {
      optional<T> element1 = generate_element<T>(1);
      optional<T> element2 = generate_element<T>(2);
      optional<T> element3 = generate_element<T>(3);
      optional<T> element4 = generate_element<T>(4);

      ds.m_union(element1, element2);
      CHECK(ds.find(element1) == ds.find(element2));

      ds.m_union(element3, element4);
      CHECK(ds.find(element3) == ds.find(element4));

      ds.m_union(element1, element3);
      CHECK(ds.find(element1) == ds.find(element3));
      CHECK(ds.find(element2) == ds.find(element4));
      CHECK(ds.find(element1) == ds.find(element2));
      CHECK(ds.find(element1) == ds.find(element4));
    }
  }

  TEST_CASE_TEMPLATE("DisjointSetMapping", T, int, std::string) {
    disjoint_set<int> ds;
    ds.m_union(1, 2);
    ds.m_union(3, 4);
    ds.m_union(1, 4);
    ds.m_union(5, 6);

    std::map<optional<int>, optional<int>, OptionalComparator<int>>
        expectedMapping = {{1, 4}, {2, 4}, {3, 4}, {4, 4}, {5, 6}, {6, 6}};

    std::map<optional<int>, optional<int>, OptionalComparator<int>> mapping =
        ds.get_mapping();

    for (auto const &kv : mapping) {
      CHECK(*kv.second == *expectedMapping[kv.first]); // Compare the values
                                                       // inside the optionals
    }
  }
}
