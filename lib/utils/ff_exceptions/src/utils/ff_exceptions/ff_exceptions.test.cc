#include "utils/ff_exceptions/mk_runtime_error.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mk_runtime_error(std::string const &)") {
    auto message = some<std::string>();
    auto err = mk_runtime_error(message);
    CHECK(err.what() == message);
  }
}
