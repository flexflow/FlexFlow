#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

std::optional<MachineMappingResult>
    MachineMappingCache::load(MachineMappingState const &state) const {
  return try_at(this->cache, state);
}

void MachineMappingCache::save(MachineMappingState const &state,
                               MachineMappingResult const &result) {
  assert(!contains_key(cache, state));
  cache.emplace(state, result);
}

} // namespace FlexFlow
