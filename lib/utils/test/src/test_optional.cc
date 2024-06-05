#include "test/utils/doctest.h"
#include "utils/optional.h"
#include <rapidcheck.h>

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE_TEMPLATE("RC arbitrary", T, int, double, char) {
    CHECK(rc::check("generate", [](std::optional<T> o) {}));
  }
}
