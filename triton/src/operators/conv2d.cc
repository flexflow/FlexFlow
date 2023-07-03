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

#include "conv2d.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Conv2D::Conv2D(
    LegionModelState* model, const LayerStrategy* strategy, size_t inChannels,
    size_t outChannels, size_t kernelH, size_t kernelW, size_t strideH,
    size_t strideW, size_t paddingH, size_t paddingW, ActivationMode act,
    size_t gps, bool bias, const char* name)
    : Operator(
          model, strategy, OP_CONV2D, name, 1 /*inputs*/,
          bias ? 2 : 1 /*weights*/, 1 /*outputs*/),
      activation(act), in_channels(inChannels), out_channels(outChannels),
      kernel_h(kernelH), kernel_w(kernelW), stride_h(strideH),
      stride_w(strideW), padding_h(paddingH), padding_w(paddingW), groups(gps),
      use_bias(bias)
{
  assert(strategy->nDims == 4);
  // We don't support partitioning over the channel dimension right now
  assert(strategy->dim[1] == 1);
}

Conv2D::~Conv2D(void) {}

void
Conv2D::Configure(Tensor* input, Weights* wts, Tensor* output, Weights* bias)
{
  assert(input != nullptr);
  assert(in_channels == input->bounds[1]);
  assert(wts != nullptr);
  assert(output != nullptr);
  if (use_bias)
    assert(bias != nullptr);
  else
    assert(bias == nullptr);
  inputs.push_back(input);
  outputs.push_back(output);
  weights.push_back(wts);
  if (use_bias)
    weights.push_back(bias);
  // Compute the input transform and extent based on the bounds and our strategy
  Point<4> tiles;
  // Compute the default chunk sizes along each dimension
  for (int i = 0; i < 4; i++)
    tiles[i] = (input->bounds[i] + strategy->dim[i] - 1) / strategy->dim[i];
  Transform<4, 4> transform;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (i == j)
        transform[i][j] = tiles[i];
      else
        transform[i][j] = 0;
  input_transform = transform;
  Rect<4> extent;
  for (int i = 0; i < 4; i++) {
    if (i < 2) {
      extent.lo[i] = 0;
      extent.hi[i] = tiles[i] - 1;
    } else {
      // Compute the ghost boundaries on the height/width dimensions
      coord_t ghost_cells = ((i < 2) ? kernel_w : kernel_h) / 2;
      extent.lo[i] = -ghost_cells;
      extent.hi[i] = tiles[i] + ghost_cells - 1;
    }
  }
  input_extent = extent;
}

Rect<4>
Conv2D::GetInputBounds(Processor proc)
{
  const Point<4> point = strategy->find_local_point(proc);
  const Transform<4, 4> transform = input_transform;
  const Point<4> offset = transform * point;
  const Rect<4> extent = input_extent;
  const Rect<4> result(extent.lo + offset, extent.hi + offset);
  Rect<4> bounds;
  for (int d = 0; d < 4; d++) {
    bounds.lo[d] = 0;
    bounds.hi[d] = inputs[0]->bounds[d] - 1;
  }
  return result.intersection(bounds);
}

Rect<4>
Conv2D::GetOutputBounds(Processor proc)
{
  DomainPoint lo, hi;
  lo.dim = 4;
  hi.dim = 4;
  for (int d = 0; d < 4; d++) {
    lo[d] = 0;
    hi[d] = outputs[0]->bounds[d] - 1;
  }
  const Domain global(lo, hi);
  const Rect<4> result = strategy->find_local_domain(proc, global);
  return result;
}

Rect<4>
Conv2D::GetWeightBounds(Realm::Processor proc)
{
  // Bounds for weight is similar to output, except the first dimension
  // wouldn't be partitioned. Note that weight is in shape
  // (out_channel, in_channel / groups, kernel_h, kernel_w)
  DomainPoint lo, hi;
  lo.dim = 4;
  hi.dim = 4;
  for (int d = 0; d < 4; d++) {
    lo[d] = 0;
    hi[d] = weights[0]->bounds[d] - 1;
  }
  const Domain global(lo, hi);
  Rect<4> result = strategy->find_local_domain(proc, global);
  result.lo[0] = 0;
  result.hi[0] = weights[0]->bounds[0] - 1;
  return result;
}

Rect<1>
Conv2D::GetBiasBounds(Realm::Processor proc)
{
  // Always return the whole bias bound
  DomainPoint lo, hi;
  lo.dim = 1;
  hi.dim = 1;
  lo[0] = 0;
  hi[0] = weights[1]->bounds[0] - 1;
  return Rect<1>(lo, hi);
}

void
Conv2D::Load(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
  Conv2DArgs& proc_args = args[local_index];
  proc_args.owner = this;
  proc_args.local_index = local_index;
  proc_args.relu = (activation == AC_MODE_RELU);
  proc_args.use_bias = use_bias;
  const Rect<4> input = GetInputBounds(proc);
  const Rect<4> output = GetOutputBounds(proc);
  proc_args.input_bounds = input;
  proc_args.local_bounds = output;
  proc_args.input_datatype = inputs[0]->type;
  proc_args.output_datatype = outputs[0]->type;
  proc_args.filter_datatype = weights[0]->type;
  if (use_bias) {
    proc_args.bias_bounds = Rect<1>(output.lo[1], output.hi[1]);
    proc_args.bias_datatype = weights[1]->type;
  }
#ifdef LEGION_USE_CUDA
  if (proc.kind() == Processor::TOC_PROC) {
    proc_args.cudnn = model->runtime_->cudnn[local_index];

    const coord_t input_n = input.hi[0] - input.lo[0] + 1;
    const coord_t input_c = input.hi[1] - input.lo[1] + 1;
    const coord_t input_h = input.hi[2] - input.lo[2] + 1;
    const coord_t input_w = input.hi[3] - input.lo[3] + 1;
    const coord_t output_n = output.hi[0] - output.lo[0] + 1;
    const coord_t output_c = output.hi[1] - output.lo[1] + 1;
    const coord_t output_h = output.hi[2] - output.lo[2] + 1;
    const coord_t output_w = output.hi[3] - output.lo[3] + 1;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.inputTensor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        proc_args.inputTensor, CUDNN_TENSOR_NCHW,
        to_cudnn_datatype(inputs[0]->type), input_n, input_c, input_h,
        input_w));

    if (use_bias) {
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.biasTensor));
      CHECK_CUDNN(cudnnSetTensor4dDescriptor(
          proc_args.biasTensor, CUDNN_TENSOR_NCHW,
          to_cudnn_datatype(weights[0]->type), 1, output_c, 1, 1));
    }

    // Require that input_c is divisible by groups
    assert(input_c % groups == 0);
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&proc_args.filterDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        proc_args.filterDesc, to_cudnn_datatype(outputs[0]->type),
        CUDNN_TENSOR_NCHW, output_c, input_c / groups, kernel_h, kernel_w));

    // Technically this will overpad
    int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
    int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;
    if (pad_h != padding_h)
      printf("Warning: changing conv_padding_h to satisfy output_h size\n");
    if (pad_w != padding_w)
      printf("Warning: changing conv_padding_w to satisfy output_w size\n");

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&proc_args.convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        proc_args.convDesc,
        pad_h,  // padding_h,
        pad_w,  // padding_w,
        stride_h, stride_w, 1 /*upscale_x*/, 1 /*upscale_y*/,
        CUDNN_CROSS_CORRELATION, to_cudnn_datatype(outputs[0]->type)));
    if (groups != 1) {
      CHECK_CUDNN(cudnnSetConvolutionGroupCount(proc_args.convDesc, groups));
    }

    if (model->runtime_->allowTensorOpMathConversion_) {
      CHECK_CUDNN(cudnnSetConvolutionMathType(
          proc_args.convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else {
      CHECK_CUDNN(cudnnSetConvolutionMathType(
          proc_args.convDesc, CUDNN_TENSOR_OP_MATH));
    }

    int n, c, h, w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        proc_args.convDesc, proc_args.inputTensor, proc_args.filterDesc, &n, &c,
        &h, &w));
    assert(n == output_n);
    assert(c == output_c);
    assert(h == output_h);
    assert(w == output_w);

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&proc_args.outputTensor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        proc_args.outputTensor, CUDNN_TENSOR_NCHW,
        to_cudnn_datatype(outputs[0]->type), n, c, h, w));
    // select forward algorithm
    const int reqAlgCnt = 8;
    int cnt = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        proc_args.cudnn, proc_args.inputTensor, proc_args.filterDesc,
        proc_args.convDesc, proc_args.outputTensor, reqAlgCnt, &cnt,
        perfResults));
    assert(cnt > 0);
    CHECK_CUDNN(perfResults[0].status);
    // printf("forwardAlgo(%d) time(%.2lf)\n", perfResults[0].algo,
    // perfResults[0].time);
    proc_args.fwdAlgo = perfResults[0].algo;

    // figure out how much workspace size we need
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        proc_args.cudnn, proc_args.inputTensor, proc_args.filterDesc,
        proc_args.convDesc, proc_args.outputTensor, proc_args.fwdAlgo,
        &proc_args.workSpaceSize));
    if (proc_args.workSpaceSize > 0) {
      for (unsigned idx = 0; idx < MAX_NUM_INSTANCES; idx++) {
        void* workspace = nullptr;
        CHECK_CUDA(cudaMalloc(&workspace, proc_args.workSpaceSize));
        workspaces[idx][local_index] = workspace;
      }
    }

    if (proc_args.relu) {
      CHECK_CUDNN(cudnnCreateActivationDescriptor(&proc_args.actiDesc));
      CHECK_CUDNN(cudnnSetActivationDescriptor(
          proc_args.actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    }

    // Copy the filter weights down to the GPU as well
    Machine::MemoryQuery query(Machine::get_machine());
    query.only_kind(Memory::GPU_FB_MEM);
    query.best_affinity_to(proc);
    assert(query.count() > 0);
    const Memory local_fb = query.first();
    Weights* wts = weights[0];
    if ((wts->local_memory[local_index].kind() != Memory::GPU_FB_MEM) ||
        (wts->local_memory[local_index].kind() != Memory::Z_COPY_MEM)) {
      void* device_ptr;
      const size_t weights_size = sizeof_datatype(wts->type) *
                                  wts->local_bounds[local_index].get_volume();
      CHECK_CUDA(cudaMalloc(&device_ptr, weights_size));
      CHECK_CUDA(cudaMemcpy(
          device_ptr, wts->local_allocation[local_index], weights_size,
          cudaMemcpyHostToDevice));
      // Free the old allocation since we no longer need it
      std::free(wts->local_allocation[local_index]);
      wts->local_allocation[local_index] = device_ptr;
      wts->local_memory[local_index] = local_fb;
    }
    // Note we don't copy down any bias weights since they are tiny and
    // can be managed by legion very efficiently
  }
#endif
}

void
Conv2D::initialize(
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
#ifdef LEGION_USE_CUDA
  unsigned arg_index = 0;
  Conv2DArgs copy_args[MAX_LOCAL_PROCS];
#endif
  for (Domain::DomainPointIterator itr(launch_domain); itr; itr++) {
    const Processor proc = shardfn->find_proc(itr.p, launch_domain);
    if (!strategy->is_local_processor(proc))
      continue;
    const unsigned local_index = strategy->find_local_offset(proc);
#ifdef LEGION_USE_CUDA
    // Need to make a copy here of the Conv2D so we can fill in our pointer
    // without racing with other instances doing the same thing
    assert(arg_index < MAX_LOCAL_PROCS);
    Conv2DArgs& arg = copy_args[arg_index++];
    arg = args[local_index];
    arg.workSpace = workspaces[instance_index][local_index];
    values[itr.p] = TaskArgument(&arg, sizeof(args));
#else
    values[itr.p] = TaskArgument(args + local_index, sizeof(Conv2DArgs));
#endif
  }
  argmaps[instance_index] = runtime->construct_future_map(
      ctx, launch_space, values, true /*collective*/, shardfn->sharding_id);

  // Create logical regions for the weights and output data
  assert(outputs.size() == 1);
  LogicalRegion output_region = instance->create_tensor_region(outputs[0]);

  // Create logical regions for the weights
  assert(!weights.empty() && (weights.size() <= 2));
  LogicalRegion weight_region = instance->create_tensor_region(weights[0]);
  LogicalRegion bias_region = LogicalRegion::NO_REGION;
  if (use_bias)
    bias_region = instance->create_tensor_region(weights[1]);

  // Create partitions for the input regions
  assert(inputs.size() == 1);
  assert(inputs[0]->region[instance_index].exists());
  LogicalRegion input_region = inputs[0]->region[instance_index];
  IndexPartition part = instance->find_or_create_partition(
      input_region.get_index_space(), launch_space, input_transform,
      input_extent, LEGION_COMPUTE_COMPLETE_KIND);
  LogicalPartition input_part = runtime->get_logical_partition_by_tree(
      ctx, part, input_region.get_field_space(), input_region.get_tree_id());

  // Create partitions for the weights and output regions
  LogicalPartition weight_part =
      instance->find_or_create_tiled_partition(weights[0], strategy);
  LogicalPartition output_part =
      instance->find_or_create_tiled_partition(outputs[0], strategy);

  // Attach weight logical regions to the existing buffers
  IndexAttachLauncher weight_attach_launcher(
      LEGION_EXTERNAL_INSTANCE, weight_region, false /*restricted*/);
  const std::vector<FieldID> attach_field(1, FID_DATA);
  for (Domain::DomainPointIterator itr(launch_domain); itr; itr++) {
    const Processor proc = shardfn->find_proc(itr.p, launch_domain);
    if (!strategy->is_local_processor(proc))
      continue;
    const unsigned local_index = strategy->find_local_offset(proc);
    const LogicalRegion weight_lr =
        runtime->get_logical_subregion_by_color(ctx, weight_part, itr.p);
    weight_attach_launcher.attach_array_soa(
        weight_lr, weights[0]->local_allocation[local_index],
        false /*column major*/, attach_field,
        weights[0]->local_memory[local_index]);
  }
  weight_attachments[instance_index] =
      runtime->attach_external_resources(ctx, weight_attach_launcher);

  if (use_bias) {
    // Bias should have the same bounds across all the processors
    // so we just attach on the first one
    AttachLauncher bias_attach_launcher(
        LEGION_EXTERNAL_INSTANCE, bias_region, bias_region,
        false /*restricted*/, false /*mapped*/);
    bias_attach_launcher.attach_array_soa(
        weights[1]->local_allocation[0], false /*column major*/, attach_field,
        weights[1]->local_memory[0]);
    bias_attachments[instance_index] =
        runtime->attach_external_resource(ctx, bias_attach_launcher);
  }

  // Construct a launcher for running the inference task
  IndexTaskLauncher& launcher = launchers[instance_index];
  launcher = IndexTaskLauncher(
      CONV2D_TASK_ID, launch_space, TaskArgument(NULL, 0),
      ArgumentMap(argmaps[instance_index]), Predicate::TRUE_PRED,
      false /*must*/, mapper, strategy->tag);
  launcher.add_region_requirement(RegionRequirement(
      input_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
      input_region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(
      output_part, 0 /*projection id*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
      output_region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(
      weight_part, 0 /*projection id*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
      weight_region));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(
        weights[1]->region[instance_index], 0 /*projection id*/,
        LEGION_READ_ONLY, LEGION_EXCLUSIVE,
        weights[1]->region[instance_index]));
    launcher.add_field(3, FID_DATA);
  }
}

void
Conv2D::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  runtime->execute_index_space(ctx, launchers[instance_index]);
}

void
Conv2D::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  runtime->detach_external_resources(ctx, weight_attachments[instance_index]);
  if (use_bias)
    runtime->detach_external_resource(ctx, bias_attachments[instance_index]);
  argmaps[instance_index] = FutureMap();
}

void
Conv2D::Free(Processor proc)
{
  assert(proc.kind() == strategy->kind);
  // If this processor is not used for this layer there is nothing to do
  if (!strategy->is_local_processor(proc))
    return;
  const unsigned local_index = strategy->find_local_offset(proc);
#ifdef LEGION_USE_CUDA
  Conv2DArgs& proc_args = args[local_index];
  if (proc.kind() == Processor::TOC_PROC) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.inputTensor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.outputTensor));
    if (use_bias) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(proc_args.biasTensor));
    }
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(proc_args.filterDesc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(proc_args.actiDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(proc_args.convDesc));
    CHECK_CUDA(cudaFree(weights[0]->local_allocation[local_index]));
    if (use_bias) {
      std::free(weights[1]->local_allocation[local_index]);
      weights[1]->local_allocation[local_index] = nullptr;
    }
    if (proc_args.workSpaceSize > 0) {
      for (int idx = 0; idx < MAX_NUM_INSTANCES; idx++) {
        CHECK_CUDA(cudaFree(workspaces[idx][local_index]));
      }
    }
  } else
#endif
  {
    for (Weights* wts : weights) {
      std::free(wts->local_allocation[local_index]);
      wts->local_allocation[local_index] = nullptr;
    }
  }
}

/*static*/ void
Conv2D::PreregisterTaskVariants(void)
{
  {
    TaskVariantRegistrar cpu_registrar(CONV2D_TASK_ID, "Conv2D CPU");
    cpu_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    cpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_cpu>(
        cpu_registrar, "Conv2D Operator");
  }
#ifdef LEGION_USE_CUDA
  {
    TaskVariantRegistrar gpu_registrar(CONV2D_TASK_ID, "Conv2D GPU");
    gpu_registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    gpu_registrar.set_leaf();
    Runtime::preregister_task_variant<forward_gpu>(
        gpu_registrar, "Conv2D Operator");
  }
#endif
}

/*static*/ void
Conv2D::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  // TODO: implement this
  assert(false);
}

#ifdef LEGION_USE_CUDA
/*static*/ void
Conv2D::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  assert(task->local_arglen == sizeof(Conv2DArgs));
  const Conv2DArgs* args = (const Conv2DArgs*)task->local_args;
  assert(regions.size() == (3 + int(args->use_bias)));
  assert(task->regions.size() == (3 + int(args->use_bias)));

  const void* input_ptr = TensorAccessor<LEGION_READ_ONLY, 4>::access(
      args->input_datatype, args->input_bounds, regions[0]);
  void* output_ptr = TensorAccessor<LEGION_WRITE_DISCARD, 4>::access(
      args->output_datatype, args->local_bounds, regions[1]);
  const void* filter_ptr = TensorAccessor<LEGION_READ_ONLY, 4>::access(
      args->filter_datatype, args->local_bounds, regions[2]);
  const void* bias_ptr = NULL;
  if (args->use_bias)
    bias_ptr = TensorAccessor<LEGION_READ_ONLY, 1>::access(
        args->bias_datatype, args->bias_bounds, regions[3]);
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
  Conv2D::forward_kernel(args, input_ptr, output_ptr, filter_ptr, bias_ptr);
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
        "%s [Conv2D] forward time (CF) = %.2fms\n",
        args->owner->op_name.c_str(), elapsed);
  }
}

/*static*/ void
Conv2D::forward_kernel(
    const Conv2DArgs* args, const void* input_ptr, void* output_ptr,
    const void* filter_ptr, const void* bias_ptr)
{
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionForward(
      args->cudnn, &alpha, args->inputTensor, input_ptr, args->filterDesc,
      filter_ptr, args->convDesc, args->fwdAlgo, args->workSpace,
      args->workSpaceSize, &beta, args->outputTensor, output_ptr));

  if (bias_ptr != NULL) {
    CHECK_CUDNN(cudnnAddTensor(
        args->cudnn, &alpha, args->biasTensor, bias_ptr, &alpha,
        args->outputTensor, output_ptr));
  }
  if (args->relu) {
    CHECK_CUDNN(cudnnActivationForward(
        args->cudnn, args->actiDesc, &alpha, args->outputTensor, output_ptr,
        &beta, args->outputTensor, output_ptr));
  }
}
#endif

}}}  // namespace triton::backend::legion
