#include "utils/testing.h"
#include "utils/test_types/test_types.h"

using namespace FlexFlow::test_types;

TEST_CASE("HASHABLE") {
  using hashable_t = test_type_t<HASHABLE>;
  CHECK_EXISTS(std::invoke_result_t<std::hash<hashable_t>, hashable_t>);
}
