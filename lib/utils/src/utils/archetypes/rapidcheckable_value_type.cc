#include "utils/archetypes/rapidcheckable_value_type.h"

namespace FlexFlow {

template struct rapidcheckable_value_type<0>;

template std::string format_as(rapidcheckable_value_type<0> const &);

template void to_json(nlohmann::json &, rapidcheckable_value_type<0> const &);

template std::ostream &operator<<(std::ostream &,
                                  rapidcheckable_value_type<0> const &);

} // namespace FlexFlow

namespace rc {

template struct Arbitrary<::FlexFlow::rapidcheckable_value_type<0>>;

}

namespace std {

template struct hash<::FlexFlow::rapidcheckable_value_type<0>>;

}
