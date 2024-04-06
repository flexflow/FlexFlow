#ifndef _FLEXFLOW_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_MODEL_TRAINING_INSTANCE_H

#include "arg_backing.h"
#include "local_training_backing.h"
#include "op-attrs/ops/loss_functions.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/tensor_guid_t.h"
#include "profiling.h"

namespace FlexFlow {

struct LocalModelTrainingInstance {
  LocalModelTrainingInstance(
      ComputationGraph,
      Allocator,
      std::unordered_map<tensor_guid_t, GenericTensorAccessorW &> slot_mapping,
      PerDeviceFFHandle,
      EnableProfiling,
      ProfilingSettings);

  LocalTrainingBacking local_training_backing;

  void forward();
  void backward();
  void update();
  void reset_metrics();
};

} // namespace FlexFlow

#endif
