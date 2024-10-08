#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

MachineMappingCache empty_machine_mapping_cache() {
  return MachineMappingCache{{}};
}

std::optional<MachineMappingResult>
    machine_mapping_cache_load(MachineMappingCache const &cache,
                               MachineMappingState const &k) {
  return try_at(cache.raw_map, k);
}

void machine_mapping_cache_save(MachineMappingCache &cache,
                                MachineMappingState const &k,
                                MachineMappingResult const &v) {
  if (contains_key(cache.raw_map, k)) {
    throw mk_runtime_error(
        fmt::format("machine_mapping_cache_save expected key to not already "
                    "exist, but received existing key {}",
                    k));
  }

  cache.raw_map.emplace(k, v);
}

} // namespace FlexFlow
