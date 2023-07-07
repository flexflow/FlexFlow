#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_MODEL_TRAINING_INSTANCE_H

#include "pcg/computation_graph.h"
#include "pcg/optimizer.h"
#include "pcg/tensor_mapping.h"
#include "runtime_backing.h"
#include "task_spec/typed_future.h"

namespace FlexFlow {

struct ModelTrainingInstance {
public:
  ComputationGraph computation_graph;
  Optimizer optimizer;
  EnableProfiling enable_profiling;
  TrainingPCG training_pcg;
  TensorMapping tensor_map;
  RuntimeBacking runtime_backing;
};

TypedFuture<void> forward(ModelTrainingInstance const &);
TypedFuture<void> backward(ModelTrainingInstance const &);
TypedFuture<void> update(ModelTrainingInstance const &);

} // namespace FlexFlow

#endif
