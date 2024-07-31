#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_EXAMPLE_SUBSTITUTIONS_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_EXAMPLE_SUBSTITUTIONS_H

#include "op-attrs/activation.dtg.h"
#include "substitutions/substitution.dtg.h"

namespace FlexFlow {

Substitution create_replicate_linear_combine(int num_dims,
                                             int result_degree);

Substitution create_linear_relu_merge(int num_dims);


} // namespace FlexFlow

#endif
