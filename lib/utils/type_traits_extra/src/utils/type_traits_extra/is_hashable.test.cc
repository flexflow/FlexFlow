#include "testing.h"
#include "utils/type_traits_extra/is_hashable.h"

using namespace FlexFlow;
using namespace FlexFlow::test_types;

TEST_CASE("is_hashable") {
  CHECK(is_hashable_v<hashable_t, hashable_t>);
  CHECK_FALSE(is_hashable_v<none_t, none_t>);
}
