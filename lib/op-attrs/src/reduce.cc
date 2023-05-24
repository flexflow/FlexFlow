#include "op-attrs/ops/reduce.h"

namespace FlexFlow {

ReduceAttrs::ReduceAttrs(stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &_axes,
                         Op _op_type,
                         bool _keepdims)
  : axes(_axes), op_type(_op_type), keepdims(_keepdims)
{ }


}
