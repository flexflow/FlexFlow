#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "binary_op.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct GatherAttrs {
public:
  GatherAttrs() = delete;
  GatherAttrs(ff_dim_t);
public:
  ff_dim_t dim;
};

bool operator==(GatherAttrs const &, GatherAttrs const &);
bool operator<(GatherAttrs const &, GatherAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::GatherAttrs, dim);

namespace std {
template <>
struct hash<::FlexFlow::GatherAttrs> {
  size_t operator()(::FlexFlow::GatherAttrs const &) const;
};
} 

#endif 
