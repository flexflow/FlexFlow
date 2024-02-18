#include "utils/testing.h"
#include "utils/algorithms/typeclass/functor/instances/vector.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("vector_functor is correct") {
    CHECK(vector_functor_is_correct_for_input_v<opaque_input_type_t>);
  }
}t
