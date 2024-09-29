#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_DP_CACHE_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_DP_CACHE_H

#include "compiler/machine_mapping/machine_mapping_result_tree/machine_mapping_result_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_state.dtg.h"
#include "utils/optional.h"

namespace FlexFlow {

class MachineMappingCache {
public:
  MachineMappingCache() = default;

  std::optional<MachineMappingResultTree> load(MachineMappingState const &) const;
  void save(MachineMappingState const &, MachineMappingResultTree const &);

private:
  std::unordered_map<MachineMappingState, MachineMappingResultTree> cache;
};

} // namespace FlexFlow

#endif
