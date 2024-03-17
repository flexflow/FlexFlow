#include "utils/stack_containers/stack_vector/stack_vector.h"
#include "utils/testing.h"
#include <range/v3/range/concepts.hpp>

static_assert(std::ranges::forward_range<stack_vector<int, 10>>);

TEST_SUITE(FF_TEST_SUITE) {
  /* TEST_CASE("") { */
  /*   CHECK_MESSAGE(false, "TODO: "); */
  /* } */
}
