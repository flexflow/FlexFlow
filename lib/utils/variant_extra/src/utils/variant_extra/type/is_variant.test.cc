#include "utils/variant_extra/type/is_variant.h"
#include "utils/testing.h"
#include <string>
#include <variant>

struct opaque_t {};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_variant_v") {
    CHECK(is_variant_v<std::variant<>>);
    CHECK(is_variant_v<std::variant<int>>);
    CHECK(is_variant_v<std::variant<int> const>);
    CHECK(is_variant_v<std::variant<int> &>);
    CHECK(is_variant_v<std::variant<int, std::string, opaque_t>>);
    CHECK_FALSE(is_variant_v<std::variant<int> *>);
    CHECK_FALSE(is_variant_v<int>);
  }
}
