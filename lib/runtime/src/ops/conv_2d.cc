#include "conv_2d.h"
#include "kernels/conv_2d_kernels.h"
#include "layer.h"
#include "legion/legion_utilities.h"
#include "mpark/variant.hpp"
#include "task_spec.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

enum Slots {
  INPUT,
  OUTPUT,
  FILTER,
  BIAS,
  FILTER_GRAD,
  INPUT_GRAD,
  OUTPUT_GRAD,
  BIAS_GRAD,
  ATTRS,
  PROFILING,
}

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Conv2D;

Tensor FFModel::conv2d(Tensor const &input,
                       int outChannels,
                       int kernelH,
                       int kernelW,
                       int strideH,
                       int strideW,
                       int paddingH,
                       int paddingW,
                       ActiMode activation,
                       int groups,
                       bool use_bias,
                       Layer const *shared_op,
                       Initializer *kernel_initializer,
                       Initializer *bias_initializer,
                       char const *name) {
  assert(input->num_dims() == 4); /*NCHW*/

  Conv2DAttrs attrs = {outChannels,
                       kernelH,
                       kernelW,
                       strideH,
                       strideW,
                       paddingH,
                       paddingW,
                       groups,
                       activation,
                       use_bias};

  TensorShape output_shape = get_output_shape(attrs, input->get_shape());
  Tensor output = this->tensor_mgr.create(output_shape, CreateGrad::YES, conv);

  std::vector<Tensor> weights;

  TensorShape kernel_shape = get_kernel_shape(attrs, input->get_shape());
  weights.push_back(this->tensor_mgr.create(
      kernel_shape, CreateGrad::YES, kernel_initializer, CHOSEN_SYNC_TYPE));

  if (use_bias) {
    TensorShape bias_shape = get_bias_shape(attrs, input->get_shape());
    weights.push_back(this->tensor_mgr.create(
        bias_shape, CreateGrad::YES, bias_initializer, CHOSEN_SYNC_TYPE));
  }

  Layer *conv =
      this->layer_mgr.create(attrs, DT_FLOAT, name, {input}, weights, {output});

  //{
  //  int numdims = 4;
  //  int dims[MAX_TENSOR_DIM];
  //  dims[3] = input->dims[3];
  //  dims[2] = outChannels;
  //  dims[1] = 1 + (input->dims[1] + 2 * paddingH - kernelH) / strideH;
  //  dims[0] = 1 + (input->dims[0] + 2 * paddingW - kernelW) / strideW;
  //  conv->outputs[0] = create_tensor_legion_ordering(
  //      numdims, dims, DT_FLOAT, conv, 0, true /*create_grad*/);
  //}
  //{
  //  int dims[4] = {kernelW, kernelH, input->dims[2], outChannels};
  //  conv->weights[0] = create_weight_legion_ordering(4,
  //                                                   dims,
  //                                                   DT_FLOAT,
  //                                                   conv,
  //                                                   true /*create_grad*/,
  //                                                   kernel_initializer,
  //                                                   CHOSEN_SYNC_TYPE);
  //}
  // if (use_bias) {
  //  int dims[1] = {outChannels};
  //  conv->weights[1] = create_weight_legion_ordering(1,
  //                                                   dims,
  //                                                   DT_FLOAT,
  //                                                   conv,
  //                                                   true /*create_grad*/,
  //                                                   bias_initializer,
  //                                                   CHOSEN_SYNC_TYPE);
  //}
  conv->add_initializer("kernel", kernel_initializer);
  conv->add_initializer("bias", bias_initializer);
  /* layers.push_back(conv); */
  return conv->outputs[0];
}

Op *Conv2D::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  return new Conv2D(model,
                    get<Conv2DAttrs>(layer->attrs),
                    inputs,
                    layer->name,
                    false /*allocate_weights*/
  );
}

/* void Conv2DParams::mark_replica_dims( */
/*     ParallelTensorShape const &input, */
/*     ParallelDim output_dims[MAX_TENSOR_DIM], */
/*     ParallelDim kernel_dims[MAX_TENSOR_DIM], */
/*     ParallelDim bias_dims[MAX_TENSOR_DIM]) const { */
/*   if (output_dims != nullptr) { */
/*     output_dims[Conv2DOutput::REPLICA].is_replica_dim = true; */
/*   } */
/*   if (kernel_dims != nullptr) { */
/*     kernel_dims[Conv2DOutput::REPLICA].is_replica_dim = true; */
/*   } */
/*   if (bias_dims != nullptr) { */
/*     bias_dims[Conv2DBias::REPLICA_1].is_replica_dim = true; */
/*     bias_dims[Conv2DBias::REPLICA_2].is_replica_dim = true; */
/*     bias_dims[Conv2DBias::REPLICA_3].is_replica_dim = true; */
/*     bias_dims[Conv2DBias::REPLICA_4].is_replica_dim = true; */
/*   } */
/* } */

/* int Conv2DParams::output_size(ParallelTensorShape const &input, */
/*                               ParallelDim output_dims[MAX_TENSOR_DIM]) const
 * { */
/*   int input_w = input.dims[Conv2DInput::WIDTH].size; */
/*   int input_h = input.dims[Conv2DInput::HEIGHT].size; */

/*   output_dims[Conv2DOutput::SAMPLE].size =
 * input.dims[Conv2DInput::SAMPLE].size; */
/*   output_dims[Conv2DOutput::CHANNEL].size = out_channels; */
/*   output_dims[Conv2DOutput::HEIGHT].size = */
/*       1 + (input_h + 2 * padding_h - kernel_h) / stride_h; */
/*   output_dims[Conv2DOutput::WIDTH].size = */
/*       1 + (input_w + 2 * padding_w - kernel_w) / stride_w; */

/*   return input.num_dims; */
/* }; */

/* int Conv2DParams::kernel_size(ParallelTensorShape const &input, */
/*                               ParallelDim kernel_dims[MAX_TENSOR_DIM]) const
 * { */
/*   kernel_dims[Conv2DKernel::CHANNEL_OUT].size = this->out_channels; */
/*   kernel_dims[Conv2DKernel::CHANNEL_IN].size = */
/*       input.dims[Conv2DInput::CHANNEL].size / this->groups; */
/*   kernel_dims[Conv2DKernel::HEIGHT].size = */
/*       this->kernel_h * input.dims[Conv2DInput::HEIGHT].degree; */
/*   kernel_dims[Conv2DKernel::WIDTH].size = */
/*       this->kernel_w * input.dims[Conv2DInput::WIDTH].degree; */

/*   return Conv2DKernel::NUMDIM; */
/* } */

/* int Conv2DParams::bias_size(ParallelTensorShape const &input, */
/*                             ParallelDim bias_dims[MAX_TENSOR_DIM]) const { */
/*   bias_dims[Conv2DBias::CHANNEL].size = this->out_channels; */

/*   return Conv2DBias::NUMDIM; */
/* }; */

/* void Conv2DParams::solve_dims(ParallelTensorShape const &input, */
/*                               ParallelDim output_dims[MAX_TENSOR_DIM], */
/*                               int *output_ndims, */
/*                               ParallelDim kernel_dims[MAX_TENSOR_DIM], */
/*                               int *kernel_ndims, */
/*                               ParallelDim bias_dims[MAX_TENSOR_DIM], */
/*                               int *bias_ndims) const { */
/*   assert((output_dims == nullptr) == (output_ndims == nullptr)); */
/*   assert((kernel_dims == nullptr) == (kernel_ndims == nullptr)); */
/*   assert((bias_dims == nullptr) == (bias_ndims == nullptr)); */

/*   std::vector<ParallelDimMappingRecord> mapping; */
/*   Conv2D::construct_mappings(mapping, this->use_bias); */

/*   this->mark_replica_dims(input, output_dims, kernel_dims, bias_dims); */

/*   std::vector<ParallelDim *> output_dim_sets; */
/*   if (output_dims != nullptr) { */
/*     output_dim_sets.push_back(output_dims); */
/*   } */

/*   std::vector<ParallelDim *> weight_dim_sets; */
/*   if (kernel_dims != nullptr) { */
/*     weight_dim_sets.push_back(kernel_dims); */
/*   } */
/*   if (bias_dims != nullptr && this->use_bias) { */
/*     weight_dim_sets.push_back(bias_dims); */
/*   } */

/*   solve_parallel_dim_mappings( */
/*       mapping, {input.dims}, weight_dim_sets, output_dim_sets); */

/*   if (output_dims != nullptr) { */
/*     *output_ndims = this->output_size(input, output_dims); */
/*   } */
/*   if (kernel_dims != nullptr) { */
/*     *kernel_ndims = this->kernel_size(input, kernel_dims); */
/*   } */
/*   if (bias_dims != nullptr && this->use_bias) { */
/*     *bias_ndims = this->bias_size(input, bias_dims); */
/*   } */
/* } */

/*static*/
/* void Conv2D::construct_mappings(std::vector<ParallelDimMappingRecord> &out,
 */
/*                                 bool use_bias) { */
/*   Conv2D::construct_output_mappings(out); */
/*   Conv2D::construct_weight_mappings(out, use_bias); */
/* } */

/*static*/
/* void Conv2D::construct_output_mappings( */
/*     std::vector<ParallelDimMappingRecord> &out) { */
/*   Op::construct_output_parallel_dims( */
/*       out, */
/*       {{Conv2DInput::CHANNEL, */
/*         MappingOperation::REPLICATE, */
/*         Conv2DOutput::REPLICA}, */
/*        {Conv2DInput::SAMPLE, MappingOperation::PARTITION,
 * Conv2DOutput::SAMPLE}, */
/*        {Conv2DInput::REPLICA, */
/*         MappingOperation::PARTITION, */
/*         Conv2DOutput::CHANNEL}, */
/*        {Conv2DInput::HEIGHT, MappingOperation::PARTITION,
 * Conv2DOutput::HEIGHT}, */
/*        {Conv2DInput::WIDTH, MappingOperation::PARTITION,
 * Conv2DOutput::WIDTH}}); */
/* } */

/*static*/
/* void Conv2D::construct_weight_mappings( */
/*     std::vector<ParallelDimMappingRecord> &out, bool use_bias) { */
/*   Op::construct_weight_parallel_dims( */
/*       out, */
/*       { */
/*           {Conv2DInput::REPLICA, */
/*            MappingOperation::PARTITION, */
/*            Conv2DKernel::CHANNEL_OUT}, */
/*           {Conv2DInput::SAMPLE, */
/*            MappingOperation::REPLICATE, */
/*            Conv2DKernel::REPLICA}, */
/*           {Conv2DInput::CHANNEL, */
/*            MappingOperation::PARTITION, */
/*            Conv2DKernel::CHANNEL_IN}, */
/*           {Conv2DInput::HEIGHT, */
/*            MappingOperation::REPLICATE, */
/*            Conv2DKernel::HEIGHT}, // Kernel::{HEIGHT, WEIGHT} would both work
 */
/*                                   // here */
/*           {Conv2DInput::WIDTH, */
/*            MappingOperation::REPLICATE, */
/*            Conv2DKernel::WIDTH}, // same as above */
/*       }, */
/*       Conv2DInput::INDEX, */
/*       Conv2DKernel::INDEX); */

/*   if (use_bias) { */
/*     Op::construct_weight_parallel_dims( */
/*         out, */
/*         {{Conv2DInput::REPLICA, Conv2DBias::REPLICA_1}, */
/*          {Conv2DInput::SAMPLE, Conv2DBias::REPLICA_2}, */
/*          {Conv2DInput::CHANNEL, Conv2DBias::CHANNEL}, */
/*          {Conv2DInput::HEIGHT, Conv2DBias::REPLICA_3}, */
/*          {Conv2DInput::WIDTH, Conv2DBias::REPLICA_4}}, */
/*         Conv2DInput::INDEX, */
/*         Conv2DBias::INDEX); */
/*   } */
/* } */

Conv2D::Conv2D(FFModel &model,
               Conv2D const &other,
               const ParallelTensor input,
               bool allocate_weights)
    : Conv2D(model,
             other.layer_guid,
             input,
             other.out_channels,
             other.kernel_h,
             other.kernel_w,
             other.stride_h,
             other.stride_w,
             other.padding_h,
             other.padding_w,
             other.activation,
             other.groups,
             other.use_bias,
             allocate_weights,
             other.name) {}

Conv2D::Conv2D(FFModel &model,
               Conv2DAttrs const &attrs,
               std::vector<ParallelTensor> const &inputs,
               char const *name,
               bool allocate_weights)
    : Conv2D(model,
             params.layer_guid,
             input,
             params.out_channels,
             params.kernel_h,
             params.kernel_w,
             params.stride_h,
             params.stride_w,
             params.padding_h,
             params.padding_w,
             params.activation,
             params.groups,
             params.use_bias,
             allocate_weights,
             name) {}

/* bool Conv2DParams::is_valid(ParallelTensorShape const &input) const { */
/*   ParallelTensorShape output_shape, kernel_shape, bias_shape; */
/*   this->solve_dims(input, */
/*                    output_shape.dims, */
/*                    &output_shape.num_dims, */
/*                    kernel_shape.dims, */
/*                    &kernel_shape.num_dims, */
/*                    bias_shape.dims, */
/*                    &bias_shape.num_dims); */
/*   bool is_valid = true; */
/*   is_valid &= input.is_valid(); */
/*   is_valid &= output_shape.is_valid(); */
/*   is_valid &= kernel_shape.is_valid(); */
/*   if (use_bias) { */
/*     is_valid &= bias_shape.is_valid(); */
/*   } */

/*   // TODO FIXME: Currently disable parallelizing the height and width
 * dimension */
/*   if (input.dims[0].degree > 1 || input.dims[1].degree > 1) { */
/*     return false; */
/*   } */

/*   return is_valid; */
/* } */

Conv2D::Conv2D(FFModel &model,
               LayerID const &_layer_guid,
               const ParallelTensor input,
               int outChannels,
               int kernelH,
               int kernelW,
               int strideH,
               int strideW,
               int paddingH,
               int paddingW,
               ActiMode activation,
               int groups,
               bool use_bias,
               bool allocate_weights,
               char const *name)
    : Op(model,
         OP_CONV2D,
         DT_FLOAT,
         name,
         1 /*inputs*/,
         use_bias ? 2 : 1 /*weights*/,
         allocate_weights,
         1 /*outputs*/,
         input),
      in_channels(input->dims[Conv2DInput::CHANNEL].size /
                  input->dims[Conv2DInput::CHANNEL].degree),
      out_channels(outChannels), kernel_h(kernelH), kernel_w(kernelW),
      stride_h(strideH), stride_w(strideW), padding_h(paddingH),
      padding_w(paddingW), activation(activation), groups(groups),
      use_bias(use_bias) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  assert(input->num_dims == Conv2DInput::NUMDIM);
  assert(this->stride_h > 0);
  assert(this->stride_w > 0);

  ParallelDim output_dims[MAX_TENSOR_DIM], kernel_dims[MAX_TENSOR_DIM],
      bias_dims[MAX_TENSOR_DIM];
  int output_ndims, kernel_ndims, bias_ndims;

  this->construct_mappings(*this->parallel_dims_mapping, this->use_bias);
  this->get_params().solve_dims(this->inputs[0]->get_shape(),
                                output_dims,
                                &output_ndims,
                                kernel_dims,
                                &kernel_ndims,
                                bias_dims,
                                &bias_ndims);

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);

    weights[Conv2DKernel::INDEX] =
        model.create_parallel_weight_legion_ordering(kernel_ndims,
                                                     kernel_dims,
                                                     DT_FLOAT,
                                                     NULL /*owner_op*/,
                                                     true /*create_grad*/,
                                                     kernel_initializer,
                                                     CHOSEN_SYNC_TYPE);

    if (use_bias) {
      Initializer *bias_initializer = new ZeroInitializer();

      weights[Conv2DBias::INDEX] =
          model.create_parallel_weight_legion_ordering(bias_ndims,
                                                       bias_dims,
                                                       DT_FLOAT,
                                                       NULL /*owner_op*/,
                                                       true /*create_grad*/,
                                                       bias_initializer,
                                                       CHOSEN_SYNC_TYPE);
    }
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      output_ndims, output_dims, DT_FLOAT, this);

  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

static OpTaskSignature get_init_task_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<Conv2dAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT, WRITE_ONLY);
  init.add_param_slot(FILTER);
  init.add_param_slot(BIAS);
  init.add_param_grad_slot(FILTER_GRAD, WRITE_ONLY);
  init.add_input_grad_slot(INPUT_GRAD);

  return init;
}

static OpTaskSignature get_fwd_task_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<Conv2dAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT, WRITE_ONLY);
  fwd.add_param_slot(FILTER);
  fwd.add_param_slot(BIAS);

  return fwd;
}

static OpTaskSignature get_bwd_task_signature() {
  OpTaskSignature bwd(OpTaskType::BWD);

  bwd.add_arg_slot<Conv2dAttrs>(ATTRS);

  bwd.add_input_slot(INPUT);
  bwd.add_input_grad_slot(INPUT_GRAD, READ_WRITE);
  bwd.add_output_slot(OUTPUT);
  bwd.add_output_grad_slot(OUTPUT_GRAD, READ_WRITE);
  bwd.add_param_slot(FILTER);
  bwd.add_param_grad_slot(FILTER_GRAD, READ_WRITE);
  bwd.add_param_grad_slot(BIAS_GRAD, READ_WRITE);

  return bwd;
}

OpTaskBinding Conv2d::get_init_task_binding() const {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, this->attrs);
  binding.bind_arg(PROFILING, this->profiling);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(FILTER, param_tensor(0));
  binding.bind(BIAS, param_tensor(1));
  binding.bind(FILTER_GRAD, param_tensor(0).grad());
  binding.bind(INPUT_GRAD, input_tensor(0).grad());

  return binding;
}

OpTaskBinding Conv2d::get_fwd_task_binding() const {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, this->attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(FILTER, param_tensor(0));
  binding.bind(BIAS, param_tensor(1));

  return binding;
}

OpTaskBinding Conv2d::get_bwd_task_binding() const {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, this->attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(INPUT_GRAD, input_tensor(0).grad());
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind(OUTPUT_GRAD, output_tensor(0).grad());
  binding.bind(FILTER, param_tensor(0));
  binding.bind(FILTER_GRAD, param_tensor(0).grad());
  binding.bind(BIAS_GRAD, param_tensor(1).grad());

  return binding;
}

void Conv2D::init(FFModel const &ff) {
  this->execute_task(ff, CONV2D_INIT_TASK_ID, get_init_task_signature());
  // assert(check_output_input_weight_same_parallel_is());
  // parallel_is = outputs[0]->parallel_is;
  // ArgumentMap argmap;
  // Context ctx = ff.config.lg_ctx;
  // Runtime *runtime = ff.config.lg_hlr;
  // set_argumentmap_for_init(ff, argmap);
  // IndexLauncher launcher(CONV2D_INIT_TASK_ID,
  //                        parallel_is,
  //                        TaskArgument(this, sizeof(Conv2D)),
  //                        argmap,
  //                        Predicate::TRUE_PRED,
  //                        false /*must*/,
  //                        0 /*mapper_id*/,
  //                        outputs[0]->machine_view.hash());
  // launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
  //                                                   0 /*projection id*/,
  //                                                   READ_ONLY,
  //                                                   EXCLUSIVE,
  //                                                   inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  // launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
  //                                                   0 /*projection id*/,
  //                                                   WRITE_ONLY,
  //                                                   EXCLUSIVE,
  //                                                   outputs[0]->region));
  // launcher.add_field(1, FID_DATA);
  // launcher.add_region_requirement(RegionRequirement(weights[0]->part,
  //                                                   0 /*projection id*/,
  //                                                   READ_ONLY,
  //                                                   EXCLUSIVE,
  //                                                   weights[0]->region));
  // launcher.add_field(2, FID_DATA);
  // // launcher.add_region_requirement(
  // //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  // //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // // launcher.add_field(3, FID_DATA);
  // launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
  //                                                   0 /*projection id*/,
  //                                                   WRITE_ONLY,
  //                                                   EXCLUSIVE,
  //                                                   weights[0]->region_grad));
  // launcher.add_field(3, FID_DATA);
  // // launcher.add_region_requirement(
  // //     RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
  // //                       WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  // // launcher.add_field(4, FID_DATA);
  // FutureMap fm = runtime->execute_index_space(ctx, launcher);
  // fm.wait_all_results();
  // set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0]: input
  regions[1]: output
  regions[2](I): filter
  regions[3](I): bias
  regions[4](O): filter_grad
  regions[5](O): input_grad
*/
PerDeviceOpState *Conv2D::init_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  // Conv2D const *conv = (Conv2D *)task->args;
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  FFHandler handle = *((FFHandler const *)task->local_args);
  auto const &attrs = acc.get_argument<Conv2dAttrs>(ATTRS);
  bool profiling = acc.get_argument<bool>(PROFILING);
  // TensorAccessorR<float, Conv2DInput::NUMDIM> acc_input(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  // TensorAccessorW<float, Conv2DOutput::NUMDIM> acc_output(regions[1],
  //                                                         task->regions[1],
  //                                                         FID_DATA,
  //                                                         ctx,
  //                                                         runtime,
  //                                                         false
  //                                                         /*readOutput*/);
  // TensorAccessorR<float, Conv2DKernel::NUMDIM> acc_kernel(
  //     regions[2], task->regions[2], FID_DATA, ctx, runtime);
  // TensorAccessorR<float, Conv2DBias::NUMDIM> acc_bias(
  //     regions[3], task->regions[3], FID_DATA, ctx, runtime);
  // TensorAccessorW<float, Conv2DKernel::NUMDIM> acc_kernel_grad(
  //     regions[3],
  //     task->regions[3],
  //     FID_DATA,
  //     ctx,
  //     runtime,
  //     false /*readOutput*/);
  // TensorAccessorW<float, 4> acc_input_grad(
  //     regions[4], task->regions[4], FID_DATA, ctx, runtime,
  //     false/*readOutput*/);
  auto input = acc.get_tensor<READ_ONLY>(INPUT);
  auto output = acc.get_tensor<WRITE_ONLY>(OUTPUT);
  auto filter = acc.get_tensor<READ_ONLY>(FILTER);
  auto bias = acc.get_tensor<READ_ONLY>(BIAS);
  auto filter_grad = acc.get_tensor<READ_WRITE>(FILTER_GRAD);
  auto input_grad = acc.get_tensor<READ_WRITE>(INPUT_GRAD);

  Conv2DPerDeviceState *m = new Conv2DPerDeviceState(handle);
  m->relu = attrs.activation == AC_MODE_RELU;
  m->use_bias = attrs.use_bias;
  m->profiling = profiling;
  // m->trainableInputs[0] = conv->trainableInputs[0]; ??
  std::strcpy(m->op_name, attrs.name);

  int input_w = input.shape[0];
  int input_h = input.shape[1];
  int input_c = input.shape[2];
  int input_n = input.shape[3];
  int output_w = output.shape[0];
  int output_h = output.shape[1];
  int output_c = output.shape[2];
  int output_n = output.shape[3];

  printf("init conv (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n,
         input_c,
         input_h,
         input_w);
  printf("init conv (output): n(%d) c(%d) h(%d) w(%d)\n",
         output_n,
         output_c,
         output_h,
         output_w);

  // printf("convDim: padding(%d %d) stride(%d %d)\n", conv->padding_h,
  // conv->padding_w, conv->stride_h, conv->stride_w);
  int pad_h =
      ((output_h - 1) * attrs.stride_h + attrs.kernel_h - input_h + 1) / 2;
  int pad_w =
      ((output_w - 1) * attrs.stride_w + attrs.kernel_w - input_w + 1) / 2;
  if (pad_h != attrs.padding_h) {
    printf("Warning: changing conv_padding_h to satisfy output_h size\n");
  }
  if (pad_w != attrs.padding_w) {
    printf("Warning: changing conv_padding_w to satisfy output_w size\n");
  }

  init_kernel(m,
              input_w,
              input_h,
              input_c,
              input_n,
              output_w,
              output_h,
              output_c,
              output_n,
              attrs.kernel_h,
              attrs.kernel_w,
              attrs.groups,
              attrs.stride_h,
              attrs.stride_w,
              pad_h,
              pad_w,
              input.get_float_ptr(),
              output.get_float_ptr(),
              filter.get_float_ptr(),
              filter_grad.get_float_ptr());

  return m;
}

// TaskSpec Conv2D::get_tasks_spec() const {
//   OpTasksSpec spec {
//     CONV2D_INIT_TASK_ID,
//     CONV2D_FWD_TASK_ID,
//     CONV2D_BWD_TASK_ID
//   };
//   auto &fwd = spec.get_fwd();

//   fwd.add_input_slot(INPUT);
//   fwd.add_param_slot(KERNEL);
//   fwd.add_output_slot(OUTPUT);

//   auto input = spec.input_tensor(0);
//   auto kernel = spec.param_tensor(0);
//   auto bias = spec.param_tensor(1);
//   auto output = spec.output_tensor(0);

//   fwd[INPUT] = input;
//   fwd[KERNEL] = kernel;
//   if (this->use_bias) {
//     fwd[BIAS] = bias;
//   }
//   fwd[OUTPUT] = output;

//   return spec;
// }

/* TaskSpec Conv2D::get_forward_task_spec() const { */
/*   TaskSpec spec = { CONV2D_FWD_TASK_ID, Pass::FWD }; */

/*   auto input = spec.add_tensor(TensorRole::INPUT, 0); */
/*   auto kernel = spec.add_tensor(TensorRole::PARAM, 0); */
/*   auto bias = spec.add_tensor(TensorRole::BIAS, 1); */
/*   auto output = spec.add_tensor(TensorRole::OUTPUT, 0); */

/*   spec.add_input(INPUT, input); */
/*   spec.add_input(KERNEL, kernel); */

/*   if (this->use_bias) { */
/*     spec.add_input(BIAS, bias); */
/*   } */

/*   spec.add_output(OUTPUT, output); */

/*   return spec; */
/* } */

/* TaskSpec Conv2D::get_backward_task_spec() const { */
/*   TaskSpec spec = { CONV2D_BWD_TASK_ID, Pass::BWD }; */

/*   auto input = spec.add_tensor(TensorRole::INPUT, 0); */
/*   auto kernel = spec.add_tensor(TensorRole::PARAM, 0); */
/*   auto bias = spec.add_tensor(TensorRole::BIAS, 1); */
/*   auto output = spec.add_tensor(TensorRole::OUTPUT, 0); */

/*   spec.add_input(INPUT, input); */
/*   spec.add_output(INPUT_GRAD, input.grad); */
/*   spec.add_input(KERNEL, kernel); */
/*   spec.add_output(KERNEL_GRAD, kernel.grad); */

/*   if (this->use_bias) { */
/*     spec.add_input(BIAS, bias); */
/*     spec.add_output(BIAS_GRAD, bias.grad); */
/*   } */

/*   spec.add_input(OUTPUT, output); */
/*   spec.add_input(OUTPUT_GRAD, output.grad); */

/*   return spec; */
/* } */

void Conv2D::forward(FFModel const &ff) {
  this->execute_task(ff, CONV2D_FWD_TASK_ID, get_fwd_task_signature());
}

void Conv2D::backward(FFModel const &ff) {
  this->execute_task(ff, CONV2D_bWD_TASK_ID, get_bwd_task_signature());
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): filter
  regions[3](I): bias
*/
void Conv2D::forward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Conv2DPerDeviceState const *m = *((Conv2DPerDeviceState **)task->local_args);

  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto input = acc.get_tensor<READ_ONLY>(INPUT);
  auto filter = acc.get_tensor<READ_ONLY>(FILTER);
  auto bias = acc.get_tensor<READ_ONLY>(BIAS);
  auto output = acc.get_tensor<WRITE_ONLY>(OUTPUT);

  // TensorAccessorR<float, Conv2DInput::NUMDIM> acc_input(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  // TensorAccessorW<float, Conv2DOutput::NUMDIM> acc_output(regions[1],
  //                                                         task->regions[1],
  //                                                         FID_DATA,
  //                                                         ctx,
  //                                                         runtime,
  //                                                         false
  //                                                         /*readOutput*/);
  // TensorAccessorR<float, Conv2DKernel::NUMDIM> acc_kernel(
  //     regions[2], task->regions[2], FID_DATA, ctx, runtime);
  // float const *acc_bias_ptr = NULL;
  // if (m->use_bias) {
  //   TensorAccessorR<float, Conv2DBias::NUMDIM> acc_bias(
  //       regions[3], task->regions[3], FID_DATA, ctx, runtime);
  //   acc_bias_ptr = acc_bias.ptr;
  // }

  profile(forward_kernel,
          m->profiling,
          "[Conv2d] forward_time = %.2lfms\n",
          m,
          input.get_float_ptr(),
          output.get_float_ptr(),
          filter.get_float_ptr(),
          bias.get_float_ptr());
}

/*
  region(I): input
  region(I/O): input_grad (if trainableInputs[0])
  region(I): output
  region(I/O): output_grad
  region(I): filter
  region(I/O): filter_grad
  region(I/O): bias_grad (if use_bias)
*/
void Conv2D::backward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  // Conv2D* conv = (Conv2D*) task->args;
  Conv2DPerDeviceState const *m = *((Conv2DPerDeviceState **)task->local_args);
  assert(regions.size() == (5 + static_cast<size_t>(m->trainableInputs[0]) +
                            static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() ==
         (5 + static_cast<size_t>(m->trainableInputs[0]) +
          static_cast<size_t>(m->use_bias)));
  size_t rid = 0;
  TensorAccessorR<float, Conv2DInput::NUMDIM> acc_input(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  float *acc_input_grad_ptr = NULL;
  if (m->trainableInputs[0]) {
    TensorAccessorW<float, Conv2DInput::NUMDIM> acc_input_grad(
        regions[rid],
        task->regions[rid],
        FID_DATA,
        ctx,
        runtime,
        true /*readOutput*/);
    acc_input_grad_ptr = acc_input_grad.ptr;
    rid++;
  }
  TensorAccessorR<float, Conv2DOutput::NUMDIM> acc_output(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, Conv2DOutput::NUMDIM> acc_output_grad(
      regions[rid],
      task->regions[rid],
      FID_DATA,
      ctx,
      runtime,
      true /*readOutput*/);
  rid++;
  TensorAccessorR<float, Conv2DKernel::NUMDIM> acc_kernel(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, Conv2DKernel::NUMDIM> acc_kernel_grad(
      regions[rid],
      task->regions[rid],
      FID_DATA,
      ctx,
      runtime,
      true /*readOutput*/);
  rid++;
  float *acc_bias_grad_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorW<float, Conv2DBias::NUMDIM> acc_bias_grad(
        regions[rid],
        task->regions[rid],
        FID_DATA,
        ctx,
        runtime,
        true /*readOutput*/);
    acc_bias_grad_ptr = static_cast<float *>(acc_bias_grad.ptr);
    rid++;
  }
  assert(rid == regions.size());

  backward_kernel_wrapper(m,
                          acc_input.ptr,
                          acc_input_grad_ptr,
                          acc_output.ptr,
                          acc_output_grad.ptr,
                          acc_kernel.ptr,
                          acc_kernel_grad.ptr,
                          acc_bias_grad_ptr);
}

bool Conv2D::estimate_sync_cost(Simulator *sim,
                                MachineView const &view,
                                CostMetrics &cost_metrics) const {
  ParallelDim kernel_dims[MAX_TENSOR_DIM], bias_dims[MAX_TENSOR_DIM];
  int kernel_ndims, bias_ndims;

  this->get_params().solve_dims(this->inputs[0]->get_shape(),
                                nullptr,
                                nullptr,
                                kernel_dims,
                                &kernel_ndims,
                                bias_dims,
                                &bias_ndims);

  cost_metrics.sync_time =
      sim->default_estimate_sync_cost(kernel_dims, kernel_ndims, view);

  if (this->use_bias) {
    cost_metrics.sync_time +=
        sim->default_estimate_sync_cost(bias_dims, bias_ndims, view);
  }

  return true;
}

tl::optional<RecordFormatter> Conv2D::as_dot() const {
  RecordFormatter rr;
  RecordFormatter r;

  r << this->inputs[0]->get_shape().as_dot();
  r << "in_channels" << this->in_channels;
  r << "out_channels" << this->out_channels;
  r << "kernel_h" << this->kernel_h;
  r << "kernel_w" << this->kernel_w;
  r << "padding_h" << this->padding_h;
  r << "padding_w" << this->padding_w;
  r << "stride_h" << this->stride_h;
  r << "stride_w" << this->stride_w;
  r << this->outputs[0]->get_shape().as_dot();
  rr << r;

  return rr;
}

bool Conv2D::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  int input_w = sub_input.dims[0].size;
  int input_h = sub_input.dims[1].size;
  int input_c = sub_input.dims[2].size;
  int input_n = sub_input.dims[3].size;
  int output_w = sub_output.dims[0].size;
  int output_h = sub_output.dims[1].size;
  int output_c = sub_output.dims[2].size;
  int output_n = sub_output.dims[3].size;
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;

  Conv2DPerDeviceState *m = sim->conv2d_meta;
  m->relu = activation == AC_MODE_RELU;
  // require input_c is divisible by groups

  // allocate tensors in simulator
  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *weight_ptr = (float *)sim->allocate(
      (size_t)output_c * input_c * kernel_h * kernel_w / groups, DT_FLOAT);
  assert(weight_ptr != NULL);
  float *bias_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);
  cost_metrics.weights_memory += cost_metrics.total_mem_diff_from(sim->offset);

  init_kernel(m,
              input_w,
              input_h,
              input_c,
              input_n,
              output_w,
              output_h,
              output_c,
              output_n,
              kernel_h,
              kernel_w,
              groups,
              stride_h,
              stride_w,
              pad_h,
              pad_w,
              input_ptr,
              output_ptr,
              weight_ptr,
              weight_ptr, // note we reuse weight_ptr for kernel_grad_ptr here
                          // to avoid allocating another tensor
              &cost_metrics.forward_time,
              &cost_metrics.backward_time);

  log_measure.debug("[Measure Conv2D] name(%s) input(%d %d %d %d) weight(%d %d "
                    "%d %d) output(%d %d %d %d) stride(%d %d) padding(%d %d) "
                    "forward_time(%.4lf) backward_time(%.4lf)\n",
                    name,
                    input_n,
                    input_c,
                    input_h,
                    input_w,
                    output_c,
                    input_c / groups,
                    kernel_h,
                    kernel_w,
                    output_n,
                    output_c,
                    output_h,
                    output_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    cost_metrics.forward_time,
                    cost_metrics.backward_time);
  return true;
}

} // namespace FlexFlow
