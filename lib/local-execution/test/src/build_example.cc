#include "kernels/device.h"
#include "local_allocator.h"
#include "local_model_training_instance.h"
#include "local_training_backing.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer.h"
#include "pcg/tensor.h"

int const BATCH_ITERS = 100;
int const BATCH_SIZE = 64;
int const HIDDEN_SIZE = 4096;
int const OUTPUT_SIZE = 10;
int const TRAINING_EPOCHS = 20;
int const WARMUP_ITERS = 5;
int const MEASURE_ITERS = 15;

using namespace FlexFlow;

LocalModelTrainingInstance init_model_training_instance() {
  // construct computation graph
  ComputationGraphBuilder builder;
  Tensor input = {{BATCH_SIZE, HIDDEN_SIZE},
                  DataType::FLOAT,
                  std::nullopt,
                  false,
                  std::nullopt};
  tensor_guid_t input_tensor = builder.input(input);
  tensor_guid_t dense_1 =
      builder.dense(input_tensor, HIDDEN_SIZE, Activation::RELU);
  tensor_guid_t dense_2 = builder.dense(dense_1, OUTPUT_SIZE, Activation::RELU);
  tensor_guid_t softmax = builder.softmax(dense_2);

  // pre-allocated tensors
  Allocator allocator = get_local_memory_allocator();
  GenericTensorAccessorW input_tensor_backing = allocator.allocate(input);
  std::unordered_map<tensor_guid_t, GenericTensorAccessorW &>
      pre_allocated_tensors;
  pre_allocated_tensors.insert({input_tensor, input_tensor_backing});

  // arguments
  ffHandle_t dnn;
  cudnnCreate(&dnn);
  ffblasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  PerDeviceFFHandle per_device_ff_handle;
  per_device_ff_handle.dnn = dnn;
  per_device_ff_handle.blas = blas_handle;
  EnableProfiling enable_profiling = EnableProfiling::NO;
  ProfilingSettings profiling_settings = {WARMUP_ITERS, MEASURE_ITERS};

  return LocalModelTrainingInstance(builder.computation_graph,
                                    allocator,
                                    pre_allocated_tensors,
                                    per_device_ff_handle,
                                    enable_profiling,
                                    profiling_settings);
}

int main() {
  // TOOD: metrics and update
  LocalModelTrainingInstance ff_model = init_model_training_instance();
  for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
    // ff_model.reset_metrics();
    for (int iter = 0; iter < BATCH_ITERS; iter++) {
      ff_model.forward();
      ff_model.backward();
      // ff_model.update();
    }
  }
}
