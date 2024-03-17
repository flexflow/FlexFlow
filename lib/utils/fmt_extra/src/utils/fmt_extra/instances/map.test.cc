#include "utils/fmt_extra/instances/map.h"
#include "utils/fmt_extra/instances/unordered_map.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::map") {
    rc::dc_check([](std::unordered_map<int, std::string> correct) {
      std::map<int, std::string> result = {correct.begin(), correct.end()};
      CHECK(fmt::to_string(result) == fmt::to_string(correct));
    });
  }
}
