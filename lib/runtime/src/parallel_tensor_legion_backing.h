#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_LEGION_BACKING_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_LEGION_BACKING_H

#include "legion.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ParallelTensorLegionBacking : use_visitable_eq<ParallelTensorLegionBacking> {
public:
  ParallelTensorLegionBacking() = delete;
  ParallelTensorLegionBacking(Legion::IndexSpace const &parallel_is,
                              Legion::LogicalRegion const &region,
                              Legion::LogicalRegion const &region_grad,
                              Legion::LogicalPartition const &part,
                              Legion::LogicalPartition const &part_grad,
                              Legion::PhysicalRegion const &phyical_region);

public:
  Legion::IndexSpace parallel_is;
  Legion::LogicalRegion region;
  Legion::LogicalRegion region_grad;
  Legion::LogicalPartition part;
  Legion::LogicalPartition part_grad;
  Legion::PhysicalRegion physical_region;
};

}

#endif
