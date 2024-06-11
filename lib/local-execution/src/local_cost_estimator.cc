#include "local-execution/local_cost_estimator.h"
#include "kernels/device.h"
#include "local-execution/local_allocator.h"
#include "local-execution/local_training_backing.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/computation_graph_builder.h"

// from #1384
#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

// from #1384
inline void setPerDeviceFFHandle(PerDeviceFFHandle *handle) {
  cudnnCreate(&handle->dnn);
  cublasCreate(&handle->blas);
  handle->workSpaceSize = 1024 * 1024;
  cudaMalloc(&handle->workSpace, handle->workSpaceSize);
  handle->allowTensorOpMathConversion = true;
}

static std::vector<TensorAttrs>
    get_piece_attrs(std::vector<ParallelTensorAttrs> const &parallel_attrs) {
  return transform(parallel_attrs, [](ParallelTensorAttrs const &p) {
    return get_piece_attrs(p);
  });
}

static TensorAttrs get_piece_attrs(ParallelTensorAttrs const &parallel_attrs) {
  return {get_piece_shape(parallel_attrs.shape),
          parallel_attrs.initializer,
          parallel_attrs.sync_type,
          parallel_attrs.create_gradients};
}

static LayerAttrs
    get_layer_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &op) {
  assert(!is_parallel_op(op));
  return std::visit(
      [](auto &&arg) -> LayerAttrs {
        return LayerAttrs{
            ComputationGraphOpAttrs{std::forward<decltype(arg)>(arg)}};
      },
      op);
}

float LocalCostEstimator::estimate_cost(
    PCGOperatorAttrs const &op,
    std::vector<ParallelTensorShape> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    std::vector<ParallelTensorAttrs> const &outputs,
    MachineView const &mv) const {
  LayerAttrs layer_attrs = get_layer_attrs_from_pcg_op_attrs(op);

  // allocate memory for inputs
  Allocator allocator = get_local_memory_allocator();
  TensorBackingMap tensor_backing_map;
  std::vector<tensor_guid_t> input_tensor_ids;

  ComputationGraphBuilder cg_builder;
  for (ParallelTensorShape const &input : inputs) {
    TensorShape const tensor_shape = get_piece_shape(input);
    tensor_guid_t tensor_id = cg_builder.create_tensor(tensor_shape);
    GenericTensorAccessorW const tensor_backing =
        allocator.allocate_tensor(tensor_shape);
    tensor_backing_map.insert({tensor_id, tensor_backing});
    input_tensor_ids.push_back(tensor_id);
  }

  // add operator
  std::vector<tensor_guid_t> output_tensor_ids =
      cg_builder.add_layer(layer_attrs,
                           input_tensor_ids,
                           get_piece_attrs(weights),
                           get_piece_attrs(outputs));

  // construct local backing
  int warmup_iters = 0;
  int measure_iters = 1;
  PerDeviceFFHandle handle;
  setPerDeviceFFHandle(&handle);

  RuntimeArgConfig runtime_arg_config = {
      DeviceSpecific<PerDeviceFFHandle>::create(handle),
      EnableProfiling::YES,
      ProfilingSettings{warmup_iters, measure_iters}};

  LocalTrainingBacking local_backing(allocator,
                                     cg_builder.computation_graph,
                                     tensor_backing_map,
                                     runtime_arg_config);

  local_backing.execute_init();
  float fwd_time = local_backing.execute_kernel(KernelType::FWD).value();
  float bwd_time = local_backing.execute_kernel(KernelType::BWD).value();

  // NOTE: is this correct?
  return fwd_time + bwd_time;
}

float LocalCostEstimator::estimate_cost(ParallelTensorShape const &tensor_shape,
                                        MachineView const &src,
                                        MachineView const &dst) const {
  // TODO: model communication cost analytically
  // temporarily return 0

  return 0.0;
}

CostEstimator get_local_cost_estimator() {
  return CostEstimator::create<LocalCostEstimator>();
}

} // namespace FlexFlow
