#include "parallel_tensor_legion_backing.h"

using namespace Legion;

namespace FlexFlow {

ParallelTensorLegionBacking::ParallelTensorLegionBacking(
    IndexSpace const &_parallel_is,
    LogicalRegion const &_region,
    LogicalRegion const &_region_grad,
    LogicalPartition const &_part,
    LogicalPartition const &_part_grad,
    PhysicalRegion const &_physical_region)
    : parallel_is(_parallel_is), region(_region), region_grad(_region_grad),
      part(_part), part_grad(_part_grad), physical_region(_physical_region) {}

} // namespace FlexFlow
