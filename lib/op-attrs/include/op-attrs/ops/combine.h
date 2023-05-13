#ifndef _FLEXFLOW_COMBINE_ATTRS_H
#define _FLEXFLOW_COMBINE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"
#include "core.h"

namespace FlexFlow {

struct CombineAttrs : public use_visitable_cmp<CombineAttrs> {
public:
  CombineAttrs() = delete;
  CombineAttrs(ff_dim_t combine_dim, int combine_degree);
public:
  ff_dim_t combine_dim;
  int combine_degree;
};

}

VISITABLE_STRUCT(::FlexFlow::CombineAttrs, combine_dim, combine_degree);
MAKE_VISIT_HASHABLE(::FlexFlow::CombineAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<CombineAttrs>::value, "CombineAttrs must be a valid opattr (see core.h)");
}

#endif 
