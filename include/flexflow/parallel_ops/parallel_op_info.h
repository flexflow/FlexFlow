#ifndef _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H
#define _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H

#include "flexflow/ffconst.h"

namespace FlexFlow {

struct ParallelOpInfo {
  friend void swap(ParallelOpInfo &, ParallelOpInfo &);

  OperatorType op_type;
  int parallel_dim;
  int parallel_degree;
};
bool operator==(ParallelOpInfo const &, ParallelOpInfo const &);

} // namespace FlexFlow

#endif /* _FLEXFLOW_PARALLEL_OPS_PARALLEL_OP_INFO_H */
