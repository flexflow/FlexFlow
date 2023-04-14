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

  Legion::IndexSpace get_or_create_task_is(MachineView const &);
  Legion::IndexSpace get_or_create_task_is(Legion::Domain const &);
  Legion::IndexSpace get_or_create_task_is(ParallelTensorDims const &);

  Legion::IndexSpace get_task_is(MachineView const &) const;
  Legion::IndexSpace get_task_is(Legion::Domain const &) const;
  Legion::IndexSpace get_task_is(ParallelTensorDims const &) const;
private:
  LegionConfig config;
  std::map<MachineView, Legion::IndexSpace> all_task_is;
};

}

#endif
