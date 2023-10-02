#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_DROPOUT_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_DROPOUT_ATTRS_H

#include "op-attrs/ops/dropout.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1DropoutAttrs {
  req<float> rate;
  req<unsigned long long> seed;
};
FF_VISITABLE_STRUCT(V1DropoutAttrs, rate, seed);
CHECK_IS_JSONABLE(V1DropoutAttrs);

V1DropoutAttrs to_v1(DropoutAttrs const &attrs);

} // namespace FlexFlow

#endif
