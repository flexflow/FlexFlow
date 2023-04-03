#ifndef _FLEXFLOW_RUNTIME_MAKE_OPERATOR_H
#define _FLEXFLOW_RUNTIME_MAKE_OPERATOR_H

#include "op-attrs/operator_attrs.h"
/* #include "operator.h" */
/* #include "ops/aggregate.h" */
#include "operator.h"

namespace FlexFlow {

template <typename ...Ts>
std::unique_ptr<Op> make_operator(FFModel &, variant<Ts...> const &, std::vector<ParallelTensor> const &);

template <typename ...Ts>
Op *make_operator_unsafe(FFModel &, variant<Ts...> const &, std::vector<ParallelTensor> const &);

/* std::unique_ptr<Conv2D> make_operator(FFModel &, Conv2DAttrs const &, ParallelTensor const &); */


}

#endif
