#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Pool2DAttrs : use_visitable_cmp<Pool2DAttrs> {
public:
  Pool2DAttrs() = delete;
  Pool2DAttrs(int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, PoolType pool_type, ActiMode activation);
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
};

}

VISITABLE_STRUCT(::FlexFlow::Pool2DAttrs, 
                 kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, pool_type, activation);
MAKE_VISIT_HASHABLE(::FlexFlow::Pool2DAttrs);

#endif 
