#include "pcg/optimizer.h"
#include "pcg/tensor.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph_builder.h"
#include "local_model_training_instance.h"
#include "local_training_backing.h"
#include "local_allocator.h"

const int BATCH_ITERS = 100;
const int BATCH_SIZE = 64;
const int HIDDEN_SIZE = 4096;
const int OUTPUT_SIZE = 10;
const int TRAINING_EPOCHS = 20;
const double DUMMY_FP_VAL = DUMMY_FP_VAL;
const size_t GPU_MEM_SIZE = 10e9; // 10 GB

using namespace FlexFlow;

LocalModelTrainingInstance init_model_training_instance() {
  // construct computation graph
  ComputationGraphBuilder builder;
  int const dims[] = {BATCH_SIZE, HIDDEN_SIZE};
  TensorShape input_shape (dims, DataType::FLOAT);
  Tensor input_tensor = builder.create_tensor(input_shape, false);
  Tensor dense_1 = builder.dense(input_tensor, HIDDEN_SIZE);
  Tensor dense_2 = builder.dense(dense_1, OUTPUT_SIZE);
  Tensor softmax = builder.softmax(dense_2);

  // pre-allocate input tensor
  Allocator allocator = get_local_memory_allocator(GPU_MEM_SIZE);
  GenericTensorAccessorW input_tensor_backing = allocator.allocate(input_tensor);
  std::unordered_map<tensor_guid_t, GenericTensorAccessorW> pre_allocated_tensors;
  pre_allocated_tensors.insert({input_tensor, input_tensor_backing});

  // optimizer
  double lr = DUMMY_FP_VAL; double momentum = DUMMY_FP_VAL; bool nesterov = false; double weight_decay = DUMMY_FP_VAL;
  SGDOptimizer optimizer = {lr, momentum, nesterov, weight_decay};
  
  // arguments
  EnableProfiling enable_profiling = EnableProfiling::NO;
  tensor_guid_t logit_tensor = softmax;
  tensor_guid_t label_tensor = softmax; // how to get the label tensor?
  LossFunction loss_fn = LossFunction::CATEGORICAL_CROSSENTROPY;
  OtherLossAttrs loss_attrs = {loss_fn};
  std::vector<Metric> metrics = {Metric::ACCURACY, Metric::CATEGORICAL_CROSSENTROPY};
  MetricsAttrs metrics_attrs (loss_fn, metrics);

  return initialize_backing(
    builder.computation_graph,
    optimizer,
    enable_profiling,
    logit_tensor,
    label_tensor,
    loss_attrs,
    metrics_attrs,
    pre_allocated_tensors
  );
}

int main() {
  LocalModelTrainingInstance ff_model = init_model_training_instance();
  for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
    ff_model.reset_metrics();
    for (int iter = 0; iter < BATCH_ITERS; iter++) {
      ff_model.forward();
      ff_model.backward();
      ff_model.update();
    }
  }
}