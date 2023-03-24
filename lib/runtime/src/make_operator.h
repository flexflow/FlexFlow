#ifndef _FLEXFLOW_RUNTIME_MAKE_OPERATOR_H
#define _FLEXFLOW_RUNTIME_MAKE_OPERATOR_H

#include "op-attrs/operator_attrs.h"
#include "operator.h"
#include "ops/aggregate.h"

namespace FlexFlow {

std::unique_ptr<Op> make_operator(PCGOperatorAttrs const &);
std::unique_ptr<Aggregate> make_operator(AggregateAttrs const &, std::vector<ParallelTensor> const &);
std::unique_ptr<AggregateSpecAttrs> make_operator(AggregateSpecAttrs const &);

}

#endif
