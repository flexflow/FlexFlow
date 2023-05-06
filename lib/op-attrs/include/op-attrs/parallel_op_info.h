#ifndef _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H
#define _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H

#include "op-attrs/ffconst.h"
#include <tuple>
#include <functional>
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct ParallelOpInfo : use_visitable_cmp<ParallelOpInfo> {
public:
  ParallelOpInfo() = delete;
  ParallelOpInfo(OperatorType op_type, ff_dim_t parallel_dim, int parallel_degree); 
public:
  OperatorType op_type;
  ff_dim_t parallel_dim;
  int parallel_degree;
};

}

VISITABLE_STRUCT(::FlexFlow::ParallelOpInfo, op_type, parallel_dim, parallel_degree);
MAKE_VISIT_HASHABLE(::FlexFlow::ParallelOpInfo);

#endif
