#include "utils/type_traits_extra/is_well_behaved_value_type.h"
#include "utils/testing.h"
#include <memory>

struct example_well_behaved_value_type {
  bool operator==(example_well_behaved_value_type const &) const;
  bool operator!=(example_well_behaved_value_type const &) const;
};

namespace std {
template <>
struct hash<example_well_behaved_value_type> {
  size_t operator()(example_well_behaved_value_type const &) const;
};
} // namespace std

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_well_behaved_value_type_v") {
    CHECK(is_well_behaved_value_type_v<example_well_behaved_value_type>);
    CHECK_FALSE(is_well_behaved_value_type_v<std::unique_ptr<int>>);
  }
}
