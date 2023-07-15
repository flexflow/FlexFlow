#ifndef _FLEXFLOW_RUNTIME_SRC_GET_DATA_DEPENDENCIES_H
#define _FLEXFLOW_RUNTIME_SRC_GET_DATA_DEPENDENCIES_H

#include "op-attrs/operator_attrs.h"
#include "task_spec/task_signature.h"

namespace FlexFlow {

struct DataCoord {
  slot_id slot;
  std::vector<int> coord;
};

struct DataDependencies {
  template <typename F>
  void add_data_dependency(slot_id, slot_id, F const &) {}
};

DataDependencies
    pointwise_data_dependencies(std::vector<slot_id> const &input_slots,
                                std::vector<slot_id> const &weight_slots,
                                std::vector<slot_id> const &output_slots);

} // namespace FlexFlow

#endif
