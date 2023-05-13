#ifndef _FLEXFLOW_CAST_ATTRS_H
#define _FLEXFLOW_CAST_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "core.h"

namespace FlexFlow {

struct CastAttrs {
public:
  CastAttrs() = delete;
  CastAttrs(DataType);
public:
  DataType dtype;
};

bool operator==(CastAttrs const &, CastAttrs const &);
bool operator!=(CastAttrs const &, CastAttrs const &);
bool operator<(CastAttrs const &, CastAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::CastAttrs, dtype);
MAKE_VISIT_HASHABLE(::FlexFlow::CastAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<CastAttrs>::value, "CastAttrs must be a valid opattr (see core.h)");
}

#endif 
