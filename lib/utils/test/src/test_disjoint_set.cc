#include "doctest.h"
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

TEST_CASE_TEMPLATE("DisjointSetUnionAndFind", T, int, std::string) {
  FlexFlow::disjoint_set<T> ds;

  SUBCASE("SingleElementSets") {
    T element = generate_element<T>(1);
    CHECK_EQ(ds.find(element), element);

    element = generate_element<T>(2);
    CHECK_EQ(ds.find(element), element);
  }

  SUBCASE("UnionAndFind") {
    T element1 = generate_element<T>(1);
    T element2 = generate_element<T>(2);
    T element3 = generate_element<T>(3);
    T element4 = generate_element<T>(4);

    ds.m_union(element1, element2);
    CHECK_EQ(ds.find(element1), ds.find(element2));

    ds.m_union(element3, element4);
    CHECK_EQ(ds.find(element3), ds.find(element4));

    ds.m_union(element1, element3);
    CHECK_EQ(ds.find(element1), ds.find(element3));
    CHECK_EQ(ds.find(element2), ds.find(element4));
    CHECK_EQ(ds.find(element1), ds.find(element2));
    CHECK_EQ(ds.find(element1), ds.find(element4));
  }
}

TEST_CASE_TEMPLATE("DisjointSetMapping", T, int, std::string) {
  disjoint_set<T> ds;

  T element1 = generate_element<T>(1);
  T element2 = generate_element<T>(2);
  T element3 = generate_element<T>(3);
  T element4 = generate_element<T>(4);

  ds.m_union(element1, element2);
  ds.m_union(element3, element4);
  ds.m_union(element1, element3);

  std::map<T, T> expectedMapping = {{element1, ds.find(element1)},
                                    {element2, ds.find(element2)},
                                    {element3, ds.find(element3)},
                                    {element4, ds.find(element4)}};
  std::map<T, T> mapping = ds.get_mapping();
  CHECK_EQ(mapping, expectedMapping);
}
