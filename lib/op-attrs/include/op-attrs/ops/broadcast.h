#ifndef _FLEXFLOW_INCLUDE_OPATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_INCLUDE_OPATTRS_OPS_BROADCAST_H

#include "core.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BroadcastAttrs {
public:
  BroadcastAttrs(stack_vector<int, MAX_TENSOR_DIM> const &);

public:
  stack_vector<int, MAX_TENSOR_DIM> target_dims;
};

bool operator==(BroadcastAttrs const &, BroadcastAttrs const &);
bool operator!=(BroadcastAttrs const &, BroadcastAttrs const &);
bool operator<(BroadcastAttrs const &, BroadcastAttrs const &);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::BroadcastAttrs, target_dims);

namespace std {
template <>
struct hash<::FlexFlow::BroadcastAttrs> {
  size_t operator()(::FlexFlow::BroadcastAttrs const &) const;
};
} // namespace std

namespace FlexFlow {

static_assert(is_valid_opattr<BroadcastAttrs>::value,
              "BroadcastAttrs must be a valid opattr (see core.h)");

}

#endif
