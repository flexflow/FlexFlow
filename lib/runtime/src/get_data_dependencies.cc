#include "get_data_dependencies.h"

namespace FlexFlow {

static void add_pointwise_data_dependency(DataDependencies &deps,
                                          slot_id input,
                                          slot_id output) {
  deps.add_data_dependency(
      input, output, [](std::vector<int> const &coords) { return coords; });
}

DataDependencies
    pointwise_data_dependence(std::vector<slot_id> const &input_slots,
                              std::vector<slot_id> const &weight_slots,
                              std::vector<slot_id> const &output_slots) {
  DataDependencies deps;
  for (slot_id output_slot : output_slots) {
    for (slot_id input_slot : input_slots) {
      add_pointwise_data_dependency(deps, input_slot, output_slot);
    }
    for (slot_id weight_slot : weight_slots) {
      add_pointwise_data_dependency(deps, weight_slot, output_slot);
    }
  }
  return deps;
}

} // namespace FlexFlow
