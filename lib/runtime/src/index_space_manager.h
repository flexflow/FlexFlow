#ifndef _FLEXFLOW_RUNTIME_SRC_INDEX_SPACE_MANAGER_H
#define _FLEXFLOW_RUNTIME_SRC_INDEX_SPACE_MANAGER_H

#include "pcg/machine_view.h"
#include "legion.h"
#include "runtime/config.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <map>

namespace FlexFlow {

struct IndexSpaceManager {
public:
  IndexSpaceManager() = delete;
  IndexSpaceManager(LegionConfig const &);

  Legion::IndexSpace const &at(MachineView const &) const;
  Legion::IndexSpace const &at(Legion::Domain const &) const;
private:
  LegionConfig config;
  mutable std::unordered_map<MachineView, Legion::IndexSpace> all_task_is;
};

}

#endif
