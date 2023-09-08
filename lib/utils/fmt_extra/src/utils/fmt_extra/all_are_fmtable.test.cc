#include "utils/testing.h"
#include "utils/fmt_extra/all_are_fmtable.h"

using namespace FlexFlow;

struct not_fmtable {};

TEST_CASE("all_are_fmtable") {
  REQUIRE(is_fmtable_v<int>);
  REQUIRE(is_fmtable_v<float>);
  REQUIRE_FALSE(is_fmtable_v<not_fmtable>);

  CHECK(all_are_fmtable_v<int, float>);
  CHECK_FALSE(all_are_fmtable_v<int, not_fmtable>);
}
