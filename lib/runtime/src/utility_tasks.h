#ifndef _FLEXFLOW_RUNTIME_SRC_UTILITY_TASKS_H
#define _FLEXFLOW_RUNTIME_SRC_UTILITY_TASKS_H

#include "kernels/ff_handle.h"
#include "legion.h"
#include <vector>

namespace FlexFlow {

class UtilityTasks {
public:
  static FFHandler
      init_cuda_task(Legion::Task const *task,
                     std::vector<Legion::PhysicalRegion> const &regions,
                     Legion::Context ctx,
                     Legion::Runtime *runtime);
  static void dummy_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  static void
      init_images_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      init_labels_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      load_images_task(Legion::Task const *task,
                       std::vector<Legion::PhysicalRegion> const &regions,
                       Legion::Context ctx,
                       Legion::Runtime *runtime);
  static void
      normalize_images_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
};

} // namespace FlexFlow

#endif
