#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_MAPPING_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_MAPPING_H

#include "initializer.h"
#include "legion.h"
#include "legion_backing.h"
#include "operator.h"
#include "runtime/config.h"
#include <string>

namespace FlexFlow {

// TODO FIXME @lockshaw modernize

void map_weight(ParallelTensor &weight,
                Op const *op,
                LegionConfig const &config);

void create_aliased_partition(int num_dims,
                              const ParallelDim dims[],
                              int aliased_dim,
                              Legion::IndexSpace const &part_is,
                              Legion::LogicalRegion const &region,
                              Legion::LogicalPartition &part,
                              LegionConfig const &config);

void create_disjoint_partition(int num_dims,
                               const ParallelDim dims[],
                               Legion::IndexSpace const &part_is,
                               Legion::LogicalRegion const &region,
                               Legion::LogicalPartition &part,
                               LegionConfig const &config);

void map_tensor(ParallelTensor &tensor,
                Op const *op,
                LegionConfig const &,
                IndexSpaceManager &);

} // namespace FlexFlow

#endif
