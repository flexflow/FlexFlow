#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct LinearAttrs : public use_visitable_cmp<LinearAttrs> {
public:
  LinearAttrs(int out_channels, bool use_bias, DataType data_type, ActiMode activation);
public:
  int out_channels;
  bool use_bias;
  DataType data_type;
  ActiMode activation;
};

}

VISITABLE_STRUCT(::FlexFlow::LinearAttrs, out_channels, use_bias, data_type, activation);
MAKE_VISIT_HASHABLE(::FlexFlow::LinearAttrs);

#endif 
