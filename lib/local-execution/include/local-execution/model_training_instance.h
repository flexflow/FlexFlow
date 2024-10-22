#ifndef _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H

#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct ModelTrainingInstance {
  ModelTrainingInstance(Allocator const &,
                        ComputationGraph const &,
                        TensorBackingMap const &,
                        RuntimeArgConfig const &,
                        LossAttrs const &,
                        tensor_guid_t const & logit_tensor,
                        tensor_guid_t const & label_tensor,
                        OptimizerAttrs const &);

  void register_and_allocate_layers();
  void allocate_optimizer_tensors();
  void execute_init();
  PerLayerElapsedTime execute_forward();
  PerLayerElapsedTime execute_backward();
  void execute_update();

  ComputationGraph computation_graph;
  LocalTrainingBacking training_backing;
  LossAttrs loss_attrs;
  tensor_guid_t logit_tensor;
  tensor_guid_t label_tensor;
  OptimizerAttrs optimizer_attrs;
};

}

#endif
