#include "utils/containers/transform_ref.h"
#include "utils/archetypes/value_type.h"
#include <functional>

namespace FlexFlow {

using T = value_type<0>;
using F = std::function<T(T const &)>;

template void transform_ref(T &, F &&);

} // namespace FlexFlow
