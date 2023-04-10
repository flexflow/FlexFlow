#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "binary_op.h"
#include "op-attrs/legion_dim.h"

namespace FlexFlow {

struct GatherAttrs {
public:
  GatherAttrs() = delete;
  GatherAttrs(legion_dim_t);
public:
  legion_dim_t legion_dim;
};

bool operator==(GatherAttrs const &, GatherAttrs const &);
bool operator<(GatherAttrs const &, GatherAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::GatherAttrs, legion_dim);

namespace std {
template <>
struct hash<::FlexFlow::GatherAttrs> {
  size_t operator()(::FlexFlow::GatherAttrs const &) const;
};
} 

#endif 
