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

#include "unary.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

UnaryArgs::UnaryArgs(void) {}

UnaryOperator::UnaryOperator(
    LegionModelState* model, const LayerStrategy* strategy, OperatorType type,
    const void* scalar_value, DataType stype, bool inplace, const char* name)
    : Operator(model, strategy, type, name, 1, 0, 1), scalar_type(stype),
      inplace(inplace)
{
  switch (stype) {
    case DT_NONE:
      break;
    case DT_INT8: {
      memcpy(&scalar.int8_value, scalar_value, sizeof(uint8_t));
      break;
    }
    case DT_HALF: {
      memcpy(&scalar.half_value, scalar_value, sizeof(__half));
      break;
    }
    case DT_FLOAT: {
      memcpy(&scalar.float_value, scalar_value, sizeof(float));
      break;
    }
    case DT_DOUBLE: {
      memcpy(&scalar.double_value, scalar_value, sizeof(double));
      break;
    }
    default:
      abort();
  }
}

UnaryOperator::~UnaryOperator(void) {}

void
UnaryOperator::Configure(Tensor* input, Tensor* output)
{
  assert(input != nullptr);
  assert(output != nullptr);
  assert(input->type == scalar_type);
  assert((op_type == OP_CAST) || (input->type == output->type));
  assert(!inplace || (input == output));
  // Make sure that they have the same bounds
  assert(input->bounds.size() == output->bounds.size());
  for (unsigned idx = 0; idx < input->bounds.size(); idx++)
    assert(input->bounds[idx] == output->bounds[idx]);
  inputs.push_back(input);
  outputs.push_back(output);
}

Domain
UnaryOperator::GetBounds(Processor proc)
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
UnaryOperator::Load(Realm::Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  UnaryArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.local_index = local_index;
  proc_args.op_type = op_type;
  proc_args.bounds = GetBounds(proc);
  proc_args.datatype = scalar_type;
  proc_args.casttype = outputs[0]->type;
  proc_args.inplace = inplace;
  switch (scalar_type) {
    case DT_NONE:
      break;
    case DT_INT8: {
      proc_args.scalar.int8_value = scalar.int8_value;
      break;
    }
    case DT_HALF: {
      proc_args.scalar.half_value = scalar.half_value;
      break;
    }
    case DT_FLOAT: {
      proc_args.scalar.float_value = scalar.float_value;
      break;
    }
    case DT_DOUBLE: {
      proc_args.scalar.double_value = scalar.double_value;
      break;
    }
    default:
      abort();
  }
#ifdef LEGION_USE_CUDA
  if (proc.kind() == Processor::TOC_PROC) {
    if (use_cudnn(op_type)) {
      proc_args.cudnn = model->runtime_->cudnn[local_index];
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.inputTensor));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.outputTensor));
      CHECK_CUDNN(cudnnCreateActivationDescriptor(&proc_args.actiDesc));
      cudnnActivationMode_t mode;
      switch (op_type) {
        case OP_SIGMOID: {
          mode = CUDNN_ACTIVATION_SIGMOID;
          break;
        }
        case OP_RELU: {
          mode = CUDNN_ACTIVATION_RELU;
          break;
        }
        case OP_TANH: {
          mode = CUDNN_ACTIVATION_TANH;
          break;
        }
        case OP_ELU: {
          mode = CUDNN_ACTIVATION_ELU;
          break;
        }
        default:
          abort();
      }
      CHECK_CUDNN(cudnnSetActivationDescriptor(
          proc_args.actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
      CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
          proc_args.inputTensor, proc_args.bounds, inputs[0]->type));
      CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
          proc_args.outputTensor, proc_args.bounds, outputs[0]->type));
    }
  }
#endif
}

void
UnaryOperator::initialize(
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
    values[itr.p] = TaskArgument(args + local_index, sizeof(UnaryArgs));
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      UNARY_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  LogicalRegion input_region = inputs[0]->region[instance_index];
  if (inplace) {
    LogicalPartition part =
        instance->find_or_create_tiled_partition(inputs[0], strategy);
    launcher.add_region_requirement(RegionRequirement(
        part, 0 /*projection id*/, LEGION_READ_WRITE, LEGION_EXCLUSIVE,
        input_region));
    launcher.add_field(0, FID_DATA);
  } else {
    // Create a logical region for the output data
    assert(outputs.size() == 1);
    LogicalRegion output_region = instance->create_tensor_region(outputs[0]);

    // Create partitions for the regions
    LogicalPartition input_part =
        instance->find_or_create_tiled_partition(inputs[0], strategy);
    LogicalPartition output_part =
        instance->find_or_create_tiled_partition(outputs[0], strategy);
    launcher.add_region_requirement(RegionRequirement(
        input_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
        input_region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(
        output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD,
        LEGION_EXCLUSIVE, output_region));
    launcher.add_field(1, FID_DATA);
  }
}

void
UnaryOperator::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
UnaryOperator::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  argmaps[instance_index] = FutureMap();
}

void
UnaryOperator::Free(Realm::Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
#ifdef LEGION_USE_CUDA
  if ((proc.kind() == Processor::TOC_PROC) && use_cudnn(op_type)) {
    const unsigned local_index = strategy->find_local_offset(proc);
    UnaryArgs& proc_args = args[local_index];
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.inputTensor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.outputTensor));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(proc_args.actiDesc));
  }
#endif
}

/*static*/ void
UnaryOperator::PreregisterTaskVariants(void)
{
  {
    TaskVariantRegistrar cpu_registrar(UNARY_TASK_ID, "Unary CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "Unary Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(UNARY_TASK_ID, "Unary GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "Unary Operator");
  }
#endif
}

/*static*/ void
UnaryOperator::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  // TODO: implement this
  assert(false);
}

#ifdef LEGION_USE_CUDA
/*static*/ void
UnaryOperator::forward_gpu(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
    Legion::Runtime* runtime)
{
  assert(task->local_arglen == sizeof(UnaryArgs));
  const UnaryArgs* args = (const UnaryArgs*)task->local_args;
#ifndef DISABLE_LEGION_CUDA_HIJACK
  ::cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  if (use_cudnn(args->op_type)) {
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
    assert(regions.size() == 1);
    assert(task->regions.size() == 1);
    void* inout_ptr = nullptr;
    size_t volume = 0;
    switch (args->bounds.get_dim()) {
#define DIMFUNC(DIM)                                            \
  case DIM: {                                                   \
    const Rect<DIM> bounds = args->bounds;                      \
    volume = bounds.volume();                                   \
    inout_ptr = TensorAccessor<LEGION_READ_WRITE, DIM>::access( \
        args->datatype, bounds, regions[0]);                    \
    break;                                                      \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
    forward_kernel(args, stream, inout_ptr, inout_ptr, volume);
  } else {
    assert(regions.size() == 2);
    assert(task->regions.size() == 2);
    const void* input_ptr = nullptr;
    void* output_ptr = nullptr;
    size_t volume = 0;
    switch (args->bounds.get_dim()) {
#define DIMFUNC(DIM)                                                \
  case DIM: {                                                       \
    const Rect<DIM> bounds = args->bounds;                          \
    volume = bounds.volume();                                       \
    input_ptr = TensorAccessor<LEGION_READ_ONLY, DIM>::access(      \
        args->datatype, bounds, regions[0]);                        \
    output_ptr = TensorAccessor<LEGION_WRITE_DISCARD, DIM>::access( \
        args->datatype, bounds, regions[1]);                        \
    break;                                                          \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        abort();
    }
    forward_kernel(args, stream, input_ptr, output_ptr, volume);
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
        "%s [Unary] forward time (CF) = %.2fms\n", args->owner->op_name.c_str(),
        elapsed);
  }
}

/*static*/ bool
UnaryOperator::use_cudnn(OperatorType optype)
{
  if (optype == OP_RELU)
    return true;
  if (optype == OP_SIGMOID)
    return true;
  if (optype == OP_TANH)
    return true;
  if (optype == OP_ELU)
    return true;
  return false;
}
#endif

}}}  // namespace triton::backend::legion
