#include "utils/testing.h"
#include "utils/ff_exceptions/ff_exceptions.h"

TEST_CASE("not_implemented") {
  not_implemented n{};
}

TEST_CASE("mk_runtime_error") {
  auto message = some<std::string>();
  auto err = mk_runtime_error(message);
  CHECK(err.what() == message);
}
