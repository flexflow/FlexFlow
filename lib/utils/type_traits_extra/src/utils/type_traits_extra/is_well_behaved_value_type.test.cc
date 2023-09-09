#include "utils/testing.h"
#include "utils/type_traits_extra/is_well_behaved_value_type.h"
#include <memory>

struct example_well_behaved_value_type { };

namespace std {
template <>
struct hash<example_well_behaved_value_type> {
  size_t operator()(example_well_behaved_value_type const &) const;
};
}

TEST_CASE("is_well_behaved_value_type_v") {
  CHECK(is_well_behaved_value_type_v<example_well_behaved_value_type>);
  CHECK_FALSE(is_well_behaved_value_type_v<std::unique_ptr<int>>);
}
