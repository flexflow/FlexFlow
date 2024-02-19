#include "utils/testing.h"
#include "utils/fmt_extra/instances/set.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::set") {
    rc::dc_check([](std::unordered_set<int> correct) {
      std::set<int> result = {correct.begin(), correct.end()};
      CHECK(fmt::to_string(result) == fmt::to_string(correct));
    });
  }
}
