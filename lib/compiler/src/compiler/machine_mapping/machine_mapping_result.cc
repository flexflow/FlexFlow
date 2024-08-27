#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/machine_mapping.h"

namespace FlexFlow {

MachineMappingResult sequential_combine(MachineMappingResult const &s1,
                                        MachineMappingResult const &s2) {
  return MachineMappingResult{s1.runtime + s2.runtime,
                              combine(s1.machine_mapping, s2.machine_mapping)};
}

MachineMappingResult parallel_combine(MachineMappingResult const &s1,
                                      MachineMappingResult const &s2) {
  return MachineMappingResult{std::max(s1.runtime, s2.runtime),
                              combine(s1.machine_mapping, s2.machine_mapping)};
}

MachineMappingResult get_infinity_machine_mapping_result() {
  return MachineMappingResult(
      std::numeric_limits<float>::infinity(),
      MachineMapping(std::unordered_map<Node, MachineView>{}));
}

void minimize_runtime(MachineMappingResult &m1,
                      MachineMappingResult const &m2) {
  if (m2.runtime < m1.runtime) {
    m1 = m2;
  }
}

} // namespace FlexFlow
