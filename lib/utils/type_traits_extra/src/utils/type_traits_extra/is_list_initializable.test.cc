#include "utils/testing.h"
#include "utils/type_traits_extra/is_list_initializable.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_list_initializable") {
    CHECK(is_list_initializable_v<int, int>);
  }
}
