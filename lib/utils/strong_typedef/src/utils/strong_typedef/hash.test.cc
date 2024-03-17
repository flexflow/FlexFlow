#include "utils/strong_typedef/hash.h"
#include "utils/strong_typedef/strong_typedef.h"
#include "utils/testing.h"
#include "utils/type_traits_extra/is_hashable.h"

struct example_typedef_t : public strong_typedef<example_typedef_t, int> {
  using strong_typedef::strong_typedef;
};
MAKE_TYPEDEF_HASHABLE(example_typedef_t);

struct not_hashable_typedef_t
    : public strong_typedef<not_hashable_typedef_t, int> {
  using strong_typedef::strong_typedef;
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("hash<strong_typedef<T>>") {
    CHECK(is_hashable_v<example_typedef_t>);
    CHECK_FALSE(is_hashable_v<not_hashable_typedef_t>);
  }
}
