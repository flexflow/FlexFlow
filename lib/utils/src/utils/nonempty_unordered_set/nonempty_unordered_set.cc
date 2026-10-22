#include "utils/nonempty_unordered_set/nonempty_unordered_set.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"
#include "utils/archetypes/rapidcheckable_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct nonempty_unordered_set<T>;

} // namespace FlexFlow

namespace nlohmann {

using J = ::FlexFlow::jsonable_ordered_value_type<0>;

template struct adl_serializer<::FlexFlow::nonempty_unordered_set<J>>;

} // namespace nlohmann

namespace rc {

using R = ::FlexFlow::rapidcheckable_value_type<0>;

///\cond
template Gen<::FlexFlow::nonempty_unordered_set<R>>
    Arbitrary<::FlexFlow::nonempty_unordered_set<R>>::arbitrary();
///\endcond

} // namespace rc

namespace std {

using T = ::FlexFlow::value_type<0>;

template struct hash<::FlexFlow::nonempty_unordered_set<T>>;

} // namespace std
