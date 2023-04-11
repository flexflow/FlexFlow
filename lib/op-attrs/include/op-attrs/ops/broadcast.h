#ifndef _FLEXFLOW_INCLUDE_OPATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_INCLUDE_OPATTRS_OPS_BROADCAST_H

#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BroadcastAttrs {
  BroadcastAttrs(stack_vector<int, MAX_TENSOR_DIM> const &);

  stack_vector<int, MAX_TENSOR_DIM> target_dims;
};

bool operator==(BroadcastAttrs const &, BroadcastAttrs const &);
bool operator!=(BroadcastAttrs const &, BroadcastAttrs const &);
bool operator<(BroadcastAttrs const &, BroadcastAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::BroadcastAttrs, target_dims);

namespace std {
template <>
struct hash<::FlexFlow::BroadcastAttrs> {
  size_t operator()(::FlexFlow::BroadcastAttrs const &) const;
};
}

#endif
