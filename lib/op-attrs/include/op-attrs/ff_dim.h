
#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_H

#include "rapidcheck.h"
#include "op-attrs/ff_dim.dtg.h"

namespace rc {
template <>
struct Arbitrary<FlexFlow::ff_dim_t> {
  static Gen<FlexFlow::ff_dim_t> arbitrary(){
    return gen::construct<FlexFlow::ff_dim_t>(gen::inRange<int>(0, MAX_TENSOR_DIM));
  }
};
} // namespace rc

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_H
