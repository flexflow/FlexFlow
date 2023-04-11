#ifndef _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H
#define _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "core.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct AggregateSpecAttrs {
public:
  AggregateSpecAttrs() = delete;
  AggregateSpecAttrs(int n, float lambda_bal);
public:
  int n;
  float lambda_bal;
};
bool operator==(AggregateSpecAttrs const &, AggregateSpecAttrs const &);
bool operator!=(AggregateSpecAttrs const &, AggregateSpecAttrs const &);
bool operator<(AggregateSpecAttrs const &, AggregateSpecAttrs const &);
}

VISITABLE_STRUCT(::FlexFlow::AggregateSpecAttrs, n, lambda_bal);

namespace std {
template <>
struct hash<::FlexFlow::AggregateSpecAttrs> {
  size_t operator()(::FlexFlow::AggregateSpecAttrs const &) const;
};
}

namespace FlexFlow {

static_assert(is_valid_opattr<AggregateSpecAttrs>::value, "AggregateSpecAttrs must be a valid opattr (see core.h)");

}

#endif 
