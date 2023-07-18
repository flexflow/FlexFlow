#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H

#include "utils/json.h"
#include "utils/variant.h"

namespace FlexFlow {

struct V1Conv2DAttrs {};
FF_VISITABLE_STRUCT(V1Conv2DAttrs);

static_assert(
    std::is_same<visit_as_tuple_t<V1Conv2DAttrs>, std::tuple<>>::value, "");

using V1CompGraphOperatorAttrs = variant<V1Conv2DAttrs>;
using V1PCGOperatorAttrs = variant<V1Conv2DAttrs>;

} // namespace FlexFlow

#endif
