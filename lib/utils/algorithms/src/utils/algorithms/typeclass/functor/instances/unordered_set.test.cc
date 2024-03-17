#include "utils/algorithms/typeclass/functor/instances/unordered_set.h"
#include "utils/testing.h"

template <typename A>
inline constexpr bool unordered_set_functor_is_correct_for_input_v =
    is_valid_functor_instance<default_functor_t<std::unordered_set<A>>>::value;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_set_functor is correct") {
    CHECK(unordered_set_functor_is_correct_for_input_v<int>);
  }
}
