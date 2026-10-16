#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template struct value_type<0>;

template std::string format_as(value_type<0> const &);

template void to_json(nlohmann::json &, value_type<0> const &);

template std::ostream &operator<<(std::ostream &, value_type<0> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::value_type<0>>;

}
