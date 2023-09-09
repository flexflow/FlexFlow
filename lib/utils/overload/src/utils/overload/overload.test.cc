#include "utils/testing.h"
#include "utils/overload/overload.h"
#include "utils/rapidcheck_extra/some.h"

TEST_CASE("overload") {
  auto example_functor = overload {
    [](int) { return 0; },
    [](std::string const &) { return 1; },
    [](auto const &) { return 2; }
  };

  CHECK(example_functor(some<int>()) == 0);
  CHECK(example_functor(some<std::string>()) == 1);
  CHECK(example_functor(some<std::vector<int>>()) == 2);
}
