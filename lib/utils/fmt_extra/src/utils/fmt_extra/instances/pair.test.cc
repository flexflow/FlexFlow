#include "utils/testing.h"
#include "utils/fmt_extra/instances/pair.h"
#include "utils/fmt_extra/instances/tuple.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::pair") {
    rc::dc_check([](std::pair<int, std::string> p) {
      std::tuple<int, std::string> t = {p.first, p.second};
      CHECK(fmt::to_string(p) == fmt::to_string(t));
    });
  }
}
