/* Copyright 2022 NVIDIA CORPORATION
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "binary.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

BinaryOperator::BinaryOperator(
    LegionModelState* model, const LayerStrategy* strategy, OperatorType type,
    bool inplace, const char* name)
    : Operator(model, strategy, type, name, 2, 0, 1), inplace(inplace)
{
}

void
BinaryOperator::Configure(Tensor* input0, Tensor* input1, Tensor* output)
{
  assert(input0 != nullptr);
  assert(input1 != nullptr);
  assert(output != nullptr);
  assert(input0->type == input1->type);
  assert(input0->type == output->type);
  // inplace can only be set to true in restricted op type (refer to FlexFlow)
  assert(
      !inplace ||
      ((input0 == output) && ((op_type == OperatorType::OP_EW_ADD) ||
                              (op_type == OperatorType::OP_EW_MUL))));
  // Make sure that they have the same bounds.
  // Broadcasting is currently not supported
  assert(input0->bounds.size() == input1->bounds.size());
  assert(input0->bounds.size() == output->bounds.size());
  for (unsigned idx = 0; idx < input0->bounds.size(); idx++) {
    assert(input0->bounds[idx] == input1->bounds[idx]);
    assert(input0->bounds[idx] == output->bounds[idx]);
  }
  inputs.push_back(input0);
  inputs.push_back(input1);
  outputs.push_back(output);
}

Domain
BinaryOperator::GetBounds(Processor proc)
{
  const size_t dims = outputs[0]->bounds.size();
  DomainPoint lo, hi;
  lo.dim = dims;
  hi.dim = dims;
  for (int d = 0; d < dims; d++) {
    lo[d] = 0;
    hi[d] = outputs[0]->bounds[d] - 1;
  }
  const Domain global(lo, hi);
  return strategy->find_local_domain(proc, global);
}

void
BinaryOperator::Load(Realm::Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  BinaryArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.op_type = op_type;
  proc_args.bounds = GetBounds(proc);
  proc_args.datatype = outputs[0]->type;
  proc_args.inplace = inplace;
#ifdef LEGION_USE_CUDA
  if (proc.kind() == Processor::TOC_PROC) {
    if (use_cudnn(op_type, proc_args.datatype)) {
      proc_args.cudnn = model->runtime_->cudnn[local_index];
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.input0Tensor));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.input1Tensor));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.outputTensor));
      CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&proc_args.opDesc));
      cudnnOpTensorOp_t mode;
      switch (op_type) {
        case OperatorType::OP_EW_ADD:
        case OperatorType::OP_EW_SUB: {
          mode = CUDNN_OP_TENSOR_ADD;
          break;
        }
        case OperatorType::OP_EW_MUL: {
          mode = CUDNN_OP_TENSOR_MUL;
          break;
        }
        default:
          abort();
      }
      cudnnDataType_t type = to_op_tensor_comp_type(
          proc_args.datatype, proc_args.datatype, proc_args.datatype);
      CHECK_CUDNN(cudnnSetOpTensorDescriptor(
          proc_args.opDesc, mode, type, CUDNN_PROPAGATE_NAN));
      CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
          proc_args.input0Tensor, proc_args.bounds, inputs[0]->type));
      CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
          proc_args.input1Tensor, proc_args.bounds, inputs[1]->type));
      CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
          proc_args.outputTensor, proc_args.bounds, outputs[0]->type));
    }
  }
#endif
}

void
BinaryOperator::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  const Domain launch_domain = strategy->get_launch_domain();
  // Find or create the launch space domain
  IndexSpace launch_space = instance->find_or_create_index_space(launch_domain);
  // Also get the sharding function from the strategy
  ShardingFunction* shardfn = strategy->sharding_function;
  // Construct a future map for the pass-by-value arguments
  std::map<DomainPoint, TaskArgument> values;
  for (Domain::DomainPointIterator itr(launch_domain); itr; itr++) {
    const Processor proc = shardfn->find_proc(itr.p, launch_domain);
    if (!strategy->is_local_processor(proc))
      continue;
    const unsigned local_index = strategy->find_local_offset(proc);
    values[itr.p] = TaskArgument(args + local_index, sizeof(BinaryArgs));
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      BINARY_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  LogicalRegion input0_region = inputs[0]->region[instance_index];
  LogicalRegion input1_region = inputs[1]->region[instance_index];
  LogicalPartition input0_part =
      instance->find_or_create_tiled_partition(inputs[0], strategy);
  LogicalPartition input1_part =
      instance->find_or_create_tiled_partition(inputs[1], strategy);
  if (inplace) {
    launcher.add_region_requirement(RegionRequirement(
        input0_part, 0 /*projection id*/, LEGION_READ_WRITE, LEGION_EXCLUSIVE,
        input0_region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(
        input1_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
        input1_region));
    launcher.add_field(1, FID_DATA);
  } else {
    // Create a logical region for the output data
    assert(outputs.size() == 1);
    LogicalRegion output_region = instance->create_tensor_region(outputs[0]);
    // Create partitions for the regions
    LogicalPartition output_part =
        instance->find_or_create_tiled_partition(outputs[0], strategy);

    launcher.add_region_requirement(RegionRequirement(
        output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE, output_region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(
        input0_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
        input0_region));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(
        input1_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
        input1_region));
    launcher.add_field(2, FID_DATA);
  }
}

void
BinaryOperator::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
BinaryOperator::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  argmaps[instance_index] = FutureMap();
}

void
BinaryOperator::Free(Realm::Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
#ifdef LEGION_USE_CUDA
  if ((proc.kind() == Processor::TOC_PROC) &&
      use_cudnn(op_type, outputs[0]->type)) {
    const unsigned local_index = strategy->find_local_offset(proc);
    BinaryArgs& proc_args = args[local_index];
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.input0Tensor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.input1Tensor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.outputTensor));
    CHECK_CUDNN(cudnnDestroyOpTensorDescriptor(proc_args.opDesc));
  }
#endif
}

/*static*/ void
BinaryOperator::PreregisterTaskVariants(void)
{
  {
    TaskVariantRegistrar cpu_registrar(BINARY_TASK_ID, "Binary CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "Binary Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(BINARY_TASK_ID, "Binary GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "Binary Operator");
  }
#endif
}

/*static*/ void
BinaryOperator::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  // TODO: implement this
  assert(false);
}

#ifdef LEGION_USE_CUDA
/*static*/ void
BinaryOperator::forward_gpu(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
    Legion::Runtime* runtime)
{
  assert(task->local_arglen == sizeof(BinaryArgs));
  const BinaryArgs* args = (const BinaryArgs*)task->local_args;
#ifndef DISABLE_LEGION_CUDA_HIJACK
  ::cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  if (use_cudnn(args->op_type, args->datatype)) {
    CHECK_CUDNN(cudnnSetStream(args->cudnn, stream));
  }
#endif
  ::cudaEvent_t t_start, t_end;
  if (args->profiling) {
    CHECK_CUDA(cudaEventCreate(&t_start));
    CHECK_CUDA(cudaEventCreate(&t_end));
#ifdef DISABLE_LEGION_CUDA_HIJACK
    CHECK_CUDA(cudaEventRecord(t_start));
#else
    CHECK_CUDA(cudaEventRecord(t_start, stream));
#endif
  }
  if (args->inplace) {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    void* inout_ptr = nullptr;
    const void* input1_ptr = nullptr;
    size_t volume = 0;
    switch (args->bounds.get_dim()) {
#define DIMFUNC(DIM)                                            \
  case DIM: {                                                   \
    const Rect<DIM> bounds = args->bounds;                      \
    volume = bounds.volume();                                   \
    inout_ptr = TensorAccessor<LEGION_READ_WRITE, DIM>::access( \
        args->datatype, bounds, regions[0]);                    \
    input1_ptr = TensorAccessor<LEGION_READ_ONLY, DIM>::access( \
        args->datatype, bounds, regions[1]);                    \
    break;                                                      \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
    forward_kernel(args, stream, inout_ptr, input1_ptr, inout_ptr, volume);
  } else {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
    const void* input0_ptr = nullptr;
    const void* input1_ptr = nullptr;
    void* output_ptr = nullptr;
    size_t volume = 0;
    switch (args->bounds.get_dim()) {
#define DIMFUNC(DIM)                                                \
  case DIM: {                                                       \
    const Rect<DIM> bounds = args->bounds;                          \
    volume = bounds.volume();                                       \
    output_ptr = TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->datatype, bounds, regions[0]);                        \
    input0_ptr = TensorAccessor<LEGION_READ_ONLY, DIM>::access(     \
        args->datatype, bounds, regions[1]);                        \
    input1_ptr = TensorAccessor<LEGION_READ_ONLY, DIM>::access(     \
        args->datatype, bounds, regions[2]);                        \
    break;                                                          \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
    forward_kernel(args, stream, input0_ptr, input1_ptr, output_ptr, volume);
  }
  if (args->profiling) {
#ifdef DISABLE_LEGION_CUDA_HIJACK
    CHECK_CUDA(cudaEventRecord(t_end));
#else
    CHECK_CUDA(cudaEventRecord(t_start, stream));
#endif
    CHECK_CUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    CHECK_CUDA(cudaEventDestroy(t_start));
    CHECK_CUDA(cudaEventDestroy(t_end));
    printf(
        "%s [Binary] forward time (CF) = %.2fms\n",
        args->owner->op_name.c_str(), elapsed);
  }
}

/*static*/ bool
BinaryOperator::use_cudnn(OperatorType optype, DataType dtype)
{
  if (to_op_tensor_comp_type(dtype, dtype, dtype) != CUDNN_DATA_UINT8) {
    switch (optype) {
      case OperatorType::OP_EW_ADD:
      case OperatorType::OP_EW_SUB:
      case OperatorType::OP_EW_MUL:
        return true;
      default:
        return false;
    }
  }
  return false;
}

#endif

}}}  // namespace triton::backend::legion
