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

#include "softmax.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Softmax::Softmax(
    LegionModelState* model, const LayerStrategy* strategy, unsigned dim,
    const char* name)
    : Operator(model, strategy, OperatorType::OP_SOFTMAX, name, 1, 0, 1),
      dim(dim)
{
}

void
Softmax::Configure(Tensor* input, Tensor* output)
{
  assert(input != nullptr);
  assert(output != nullptr);
  assert(input->type == output->type);
  // Make sure that they have the same bounds
  assert(input->bounds.size() == output->bounds.size());
  for (unsigned idx = 0; idx < input->bounds.size(); idx++)
    assert(input->bounds[idx] == output->bounds[idx]);
  inputs.push_back(input);
  outputs.push_back(output);
}

Domain
Softmax::GetBounds(Processor proc)
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
Softmax::Load(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  assert(inputs[0]->bounds.size() == size_t(strategy->nDims));
  // Make sure that we don't have any partitions along the dimension
  // on which we are going to perform the softmax computation
  assert(strategy->dim[dim] == 1);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  SoftmaxArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.local_index = local_index;
  proc_args.bounds = GetBounds(proc);
  proc_args.datatype = inputs[0]->type;
  proc_args.dim = dim;
#ifdef LEGION_USE_CUDA
  if (proc.kind() == Processor::TOC_PROC) {
    proc_args.cudnn = model->runtime_->cudnn[local_index];
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.inputTensor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.outputTensor));
    Domain cudnn_bounds;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    // Need to figure out how to pad the bounds to make sure the
    // softmax is done over the right dimension
    switch (proc_args.bounds.get_dim()) {
      case 1: {
        assert(dim == 0);
        DomainPoint lo, hi;
        lo.dim = 4;
        hi.dim = 4;
        for (int d = 0; d < 4; d++) {
          if (d == 1) {
            lo[d] = proc_args.bounds.lo()[0];
            hi[d] = proc_args.bounds.hi()[0];
          } else {
            lo[d] = 0;
            hi[d] = 0;
          }
        }
        cudnn_bounds = Domain(lo, hi);
        break;
      }
      case 2: {
        DomainPoint lo, hi;
        lo.dim = 4;
        hi.dim = 4;
        if (dim == 0) {
          lo[0] = 0;
          hi[0] = 0;
          for (int d = 1; d <= 2; d++) {
            lo[d] = proc_args.bounds.lo()[d - 1];
            hi[d] = proc_args.bounds.hi()[d - 1];
          }
          lo[3] = 0;
          hi[3] = 0;
        } else {
          assert(dim == 1);
          format = CUDNN_TENSOR_NHWC;
          for (int d = 0; d < 2; d++) {
            lo[d] = 0;
            hi[d] = 0;
          }
          for (int d = 2; d < 4; d++) {
            lo[d] = proc_args.bounds.lo()[d - 2];
            hi[d] = proc_args.bounds.hi()[d - 2];
          }
        }
        cudnn_bounds = Domain(lo, hi);
        break;
      }
      case 3: {
        DomainPoint lo, hi;
        lo.dim = 4;
        hi.dim = 4;
        if (dim < 2) {
          if (dim == 0) {
            lo[0] = 0;
            hi[0] = 0;
          } else {
            lo[3] = 0;
            hi[3] = 0;
          }
          for (int d = 1; d <= 3; d++) {
            lo[d - dim] = proc_args.bounds.lo()[d - 1];
            hi[d - dim] = proc_args.bounds.hi()[d - 1];
          }
        } else {
          assert(dim == 2);
          format = CUDNN_TENSOR_NHWC;
          lo[0] = 0;
          hi[0] = 0;
          for (int d = 1; d < 4; d++) {
            lo[d] = proc_args.bounds.lo()[d - 1];
            hi[d] = proc_args.bounds.hi()[d - 1];
          }
        }
        cudnn_bounds = Domain(lo, hi);
        break;
      }
      case 4: {
        if (dim == 0) {
          // cudnn claims to support this type, but apparent not
          // format = CUDNN_TENSOR_CHWN;
          // cudnn_bounds = proc_args.bounds;
          fprintf(stderr, "Unsupported cudnn softmax format");
          abort();
        } else if (dim == 1) {
          format = CUDNN_TENSOR_NCHW;
          cudnn_bounds = proc_args.bounds;
        } else if (dim == 2) {
          // There's no way to do this with cudnn even with 5-d tensors
          // given the kinds of locations for the channel dimension that
          // cudnn is willing to support
          fprintf(stderr, "Unsupported cudnn softmax format");
          abort();
        } else {
          assert(dim == 3);
          format = CUDNN_TENSOR_NHWC;
          cudnn_bounds = proc_args.bounds;
        }
        break;
      }
      default:
        assert(false);
    }
    CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
        proc_args.inputTensor, cudnn_bounds, inputs[0]->type, format));
    CHECK_CUDNN(cudnnSetTensorDescriptorFromDomain(
        proc_args.outputTensor, cudnn_bounds, outputs[0]->type, format));
  }
#endif
}

void
Softmax::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
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
    values[itr.p] = TaskArgument(args + local_index, sizeof(SoftmaxArgs));
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      SOFTMAX_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  LogicalRegion input_region = inputs[0]->region[instance_index];
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
      output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
      output_region));
  launcher.add_field(1, FID_DATA);
}

void
Softmax::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
Softmax::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  argmaps[instance_index] = FutureMap();
}

void
Softmax::Free(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
#ifdef LEGION_USE_CUDA
  if (proc.kind() == Processor::TOC_PROC) {
    const unsigned local_index = strategy->find_local_offset(proc);
    SoftmaxArgs& proc_args = args[local_index];
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.inputTensor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.outputTensor));
  }
#endif
}

/*static*/ void
Softmax::PreregisterTaskVariants(void)
{
  {
    TaskVariantRegistrar cpu_registrar(SOFTMAX_TASK_ID, "Softmax CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "Softmax Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(SOFTMAX_TASK_ID, "Softmax GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "Softmax Operator");
  }
#endif
}

/*static*/ void
Softmax::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  // TODO: implement this
  assert(false);
}

SoftmaxArgs::SoftmaxArgs(void) {}

#ifdef LEGION_USE_CUDA
/*static*/ void
Softmax::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(SoftmaxArgs));
  const SoftmaxArgs* args = (const SoftmaxArgs*)task->local_args;
#ifndef DISABLE_LEGION_CUDA_HIJACK
  ::cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDNN(cudnnSetStream(args->cudnn, stream));
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
  float alpha = 1.f, beta = 0.f;
  // TODO: can we get away with CUDNN_SOFTMAX_FAST for inference?
  CHECK_CUDNN(cudnnSoftmaxForward(
      args->cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
      args->inputTensor, input_ptr, &beta, args->outputTensor, output_ptr));
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
        "%s [Softmax] forward time (CF) = %.2fms\n",
        args->owner->op_name.c_str(), elapsed);
  }
}
#endif

}}}  // namespace triton::backend::legion
