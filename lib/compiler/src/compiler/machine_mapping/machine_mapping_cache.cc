#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

std::optional<MachineMappingResult>
    MachineMappingCache::load(MachineMappingState const &state) const {
  if (contains_key(cache, state)) {
    MachineMappingResult result = cache.at(state);
    return result;
  }
  return std::nullopt;
}

void MachineMappingCache::save(MachineMappingState const &state,
                               MachineMappingResult const &result) {
  assert(!contains_key(cache, state));
  cache.emplace(state, result);
}

} // namespace FlexFlow
