#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_MODEL_TRAINING_INSTANCE_H

#include "legion_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/tensor_mapping.h"
#include "profiling.h"
#include "task_spec/typed_future.h"
#include "training_pcg.h"

namespace FlexFlow {

struct ModelTrainingInstance {
  ComputationGraph computation_graph;
  Optimizer optimizer;
  EnableProfiling enable_profiling;
  TrainingPCG training_pcg;
  TensorMapping tensor_map;
  LegionBacking legion_backing;
};
FF_VISITABLE_STRUCT(ModelTrainingInstance,
                    computation_graph,
                    optimizer,
                    enable_profiling,
                    training_pcg,
                    tensor_map,
                    legion_backing);

TypedFuture<void> forward(ModelTrainingInstance const &);
TypedFuture<void> backward(ModelTrainingInstance const &);
TypedFuture<void> update(ModelTrainingInstance const &);

} // namespace FlexFlow

#endif
