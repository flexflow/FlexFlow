#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H

#include "utils/json.h"
#include "utils/variant.h"
#include "pcg/file_format/keyed_variant.h"

namespace FlexFlow {

struct V1Conv2DAttrs : public use_visitable_cmp<V1Conv2DAttrs> {
};

using V1CompGraphOperatorAttrsVariant = variant<V1Conv2DAttrs>;

enum class V1ComputationGraphOperatorType {
  CONV_2D = index_of_type<V1Conv2DAttrs, V1CompGraphOperatorAttrsVariant>::value
};

using V1CompGraphOperatorAttrs = KeyedVariant<V1ComputationGraphOperatorType, V1CompGraphOperatorAttrsVariant>;

using V1PCGOperatorAttrsVariant = variant<V1Conv2DAttrs>;

enum class V1PCGOperatorType {
  CONV_2D = index_of_type<V1Conv2DAttrs, V1PCGOperatorAttrsVariant>::value
};

using V1PCGOperatorAttrs = KeyedVariant<V1PCGOperatorType, V1PCGOperatorAttrsVariant>;

}

VISITABLE_STRUCT_EMPTY(::FlexFlow::V1Conv2DAttrs);
MAKE_VISIT_HASHABLE(::FlexFlow::V1Conv2DAttrs);

#endif
