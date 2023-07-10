#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct DropoutAttrs {
  req<float> rate;
  req<unsigned long long> seed;
};
FF_VISITABLE_STRUCT(DropoutAttrs, rate, seed);
CHECK_VALID_OP_ATTR(DropoutAttrs);

} // namespace FlexFlow

#endif
