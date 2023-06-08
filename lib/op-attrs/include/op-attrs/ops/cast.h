#ifndef _FLEXFLOW_CAST_ATTRS_H
#define _FLEXFLOW_CAST_ATTRS_H

#include "core.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct CastAttrs : public use_visitable_cmp<CastAttrs> {
public:
  CastAttrs() = delete;
  CastAttrs(DataType);

public:
  DataType dtype;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::CastAttrs, dtype);
MAKE_VISIT_HASHABLE(::FlexFlow::CastAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<CastAttrs>::value,
              "CastAttrs must be a valid opattr (see core.h)");
}

#endif
