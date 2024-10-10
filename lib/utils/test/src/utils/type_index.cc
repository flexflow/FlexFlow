#include "utils/type_index.h"
#include <doctest/doctest.h>
#include <typeindex>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_type_index_for_type") {
    SUBCASE("int type") {
      std::type_index idx = get_type_index_for_type<int>();
      std::type_index expected_idx = typeid(int);
      CHECK_WITHOUT_STRINGIFY(idx == expected_idx);
    }

    SUBCASE("string type") {
      std::type_index idx = get_type_index_for_type<std::string>();
      std::type_index expected_idx = typeid(std::string);
      CHECK_WITHOUT_STRINGIFY(idx == expected_idx);
    }
  }

  TEST_CASE("matches<T>(std::type_index)") {
    std::type_index idx = typeid(float);

    SUBCASE("matching type") {
      CHECK(matches<float>(idx));
    }

    SUBCASE("non-matching type") {
      CHECK_FALSE(matches<int>(idx));
    }
  }
}
