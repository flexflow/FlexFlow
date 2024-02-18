#include "utils/testing.h"
#include "utils/ff_exceptions/ff_exceptions.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mk_runtime_error(std::string const &)") {
    auto message = some<std::string>();
    auto err = mk_runtime_error(message);
    CHECK(err.what() == message);
  }
}
