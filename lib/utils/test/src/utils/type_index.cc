#include "utils/type_index.h"
#include "test/utils/doctest.h"
#include <typeindex>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_type_index_for_type") {
    SUBCASE("int type") {
      std::type_index idx = get_type_index_for_type<int>();
      std::type_index expected_idx = typeid(int);
      CHECK(idx == expected_idx);
    }

    SUBCASE("string type") {
      std::type_index idx = get_type_index_for_type<std::string>();
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
