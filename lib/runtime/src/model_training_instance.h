#ifndef _FLEXFLOW_RUNTIME_SRC_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_RUNTIME_SRC_MODEL_TRAINING_INSTANCE_H

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

} // namespace FlexFlow

#endif
