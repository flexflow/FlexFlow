#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_CACHE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_CACHE_H

#include "compiler/machine_mapping/machine_mapping_cache.dtg.h"

namespace FlexFlow {

MachineMappingCache empty_machine_mapping_cache();
std::optional<MachineMappingResult> machine_mapping_cache_load(MachineMappingCache const &, MachineMappingState const &);
void machine_mapping_cache_save(MachineMappingCache &, MachineMappingState const &, MachineMappingResult const &);

} // namespace FlexFlow

#endif
