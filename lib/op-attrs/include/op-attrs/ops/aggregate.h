#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "core.h"


namespace FlexFlow {

struct AggregateAttrs {
  AggregateAttrs() = delete;
  AggregateAttrs(int n, float lambda_bal);
public:
  int n;
  float lambda_bal;
};

bool operator==(AggregateAttrs const &, AggregateAttrs const &);
bool operator!=(AggregateAttrs const &, AggregateAttrs const &);
bool operator<(AggregateAttrs const &, AggregateAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::AggregateAttrs, n, lambda_bal);

namespace std {
template <>
struct hash<::FlexFlow::AggregateAttrs> {
  size_t operator()(::FlexFlow::AggregateAttrs const &) const;
};
}

namespace FlexFlow {

static_assert(is_valid_opattr<AggregateAttrs>::value, "AggregateAttrs must be a valid opattr (see core.h)");

}

#endif 
