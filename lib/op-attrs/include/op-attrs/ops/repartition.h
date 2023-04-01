#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct RepartitionAttrs {
public:
  int repartition_legion_dim;
  int repartition_degree;
};
bool operator==(RepartitionAttrs const &, RepartitionAttrs const &);
bool operator<(RepartitionAttrs const &, RepartitionAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::RepartitionAttrs, repartition_legion_dim, repartition_degree);

namespace std {
template <>
struct hash<::FlexFlow::RepartitionAttrs> {
  size_t operator()(::FlexFlow::RepartitionAttrs const &) const;
};
}

#endif 
