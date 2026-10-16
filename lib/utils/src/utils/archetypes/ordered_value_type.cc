#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

template struct ordered_value_type<0>;

template std::string format_as(ordered_value_type<0> const &);

template void to_json(nlohmann::json &, ordered_value_type<0> const &);

template std::ostream &operator<<(std::ostream &,
                                  ordered_value_type<0> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::ordered_value_type<0>>;

}
