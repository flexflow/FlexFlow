#include "utils/testing.h"
#include "utils/type_traits_extra/is_decayable.h"

TEST_CASE("is_decayable_v") {
  CHECK(is_decayable_v<int *>);
  CHECK(is_decayable_v<int const>);
  CHECK_FALSE(is_decayable_v<int>);
}
