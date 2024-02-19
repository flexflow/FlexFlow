#include "utils/testing.h"
#include "utils/fmt_extra/instances/list.h"
#include "utils/fmt_extra/instances/vector.h"
#include "utils/fmt_extra/is_fmtable.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::list") {
    rc::dc_check([](std::vector<int> correct) {
      std::list<int> result = {correct.begin(), correct.end()};
      CHECK(fmt::to_string(result) == fmt::to_string(correct));
    });
  }
}
