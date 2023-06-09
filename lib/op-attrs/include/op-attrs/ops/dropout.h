#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct DropoutAttrs : public use_visitable_cmp<DropoutAttrs> {
public:
  DropoutAttrs(float rate, unsigned long long seed);

public:
  float rate;
  unsigned long long seed;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::DropoutAttrs, rate, seed);
MAKE_VISIT_HASHABLE(::FlexFlow::DropoutAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<DropoutAttrs>::value,
              "DropoutAttrs must be a valid opattr (see core.h)");
}

#endif
