#include "utils/testing.h"
#include "utils/type_traits_extra/is_optional.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_optional_v") {
    CHECK(is_optional_v<std::optional<int>>);
    CHECK(is_optional_v<std::optional<int> const>);
    CHECK(is_optional_v<std::nullopt_t>);
    CHECK(!is_optional_v<int>);
    CHECK(!is_optional_v<std::optional<int> *>);
  }
}
