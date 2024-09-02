#include <doctest/doctest.h>
#include "utils/type_index.h"
#include <typeindex>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("type_index function") {
    SUBCASE("int type") {
      std::type_index idx = type_index<int>();
      std::type_index expected_idx = typeid(int);
      CHECK(idx == expected_idx);
    }

    SUBCASE("string type") {
      std::type_index idx = type_index<std::string>();
      std::type_index expected_idx = typeid(std::string);
      CHECK(idx == expected_idx);
    }
  }

  TEST_CASE("matches function") {
    std::type_index idx = typeid(float);

    SUBCASE("matching type") {
      bool result = matches<float>(idx);
      CHECK(result == true);
    }

    SUBCASE("non-matching type") {
      bool result = matches<int>(idx);
      CHECK(result == false);
    }
  }
}
