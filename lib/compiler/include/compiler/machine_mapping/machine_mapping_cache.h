#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_DP_CACHE_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_DP_CACHE_H

#include "machine_mapping_result.dtg.h"
#include "machine_mapping_state.dtg.h"
#include "utils/optional.h"

namespace FlexFlow {

class MachineMappingCache {
public:
  MachineMappingCache() = default;

  std::optional<MachineMappingResult> load(MachineMappingState const &) const;
  void save(MachineMappingState const &, MachineMappingResult const &);

private:
  std::unordered_map<MachineMappingState, MachineMappingResult> cache;
};

} // namespace FlexFlow

#endif
