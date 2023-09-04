#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "core.h"
#include "op-attrs/activation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class PoolOp {
  MAX,
  AVG,
};

std::string format_as(PoolOp);

struct Pool2DAttrs {
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  PoolOp pool_type;
  req<Activation> activation;
};
FF_VISITABLE_STRUCT(Pool2DAttrs,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    pool_type,
                    activation);
FF_VISIT_FMTABLE(Pool2DAttrs);

CHECK_VALID_OP_ATTR(Pool2DAttrs);

} // namespace FlexFlow

#endif
