#include "utils/containers/set_minus_inplace.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template void set_minus_inplace(std::set<T> &, std::set<T> const &);

} // namespace FlexFlow
