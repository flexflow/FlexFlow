#ifndef _FLEXFLOW_TOPK_ATTRS_H
#define _FLEXFLOW_TOPK_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/unary_op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct TopKAttrs {
public:
  TopKAttrs() = delete;
  TopKAttrs(int k, bool sorted);
public:
  int k;
  bool sorted;
};

bool operator==(TopKAttrs const &, TopKAttrs const &);
bool operator!=(TopKAttrs const &, TopKAttrs const &);
bool operator<(TopKAttrs const &, TopKAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::TopKAttrs, k, sorted);

namespace std {
template <>
struct hash<::FlexFlow::TopKAttrs> {
  size_t operator()(::FlexFlow::TopKAttrs const &) const;
};
}

#endif
