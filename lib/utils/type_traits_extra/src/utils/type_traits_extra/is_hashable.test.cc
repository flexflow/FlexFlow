#include "utils/type_traits_extra/is_hashable.h"
#include "utils/testing.h"

struct hashable_t {};
struct not_hashable_t {};

namespace std {
template <>
struct hash<hashable_t> {
  size_t operator()(hashable_t const &) const;
};
} // namespace std

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_hashable") {
    CHECK(is_hashable_v<hashable_t>);
    CHECK_FALSE(is_hashable_v<not_hashable_t>);
  }
}
