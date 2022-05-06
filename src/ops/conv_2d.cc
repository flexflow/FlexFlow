#include "flexflow/ops/conv_2d.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"
#include "flexflow/layer.h"
#include "flexflow/model.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::coord_t;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::InlineLauncher;

Tensor FFModel::conv2d(const Tensor input,
                       int outChannels,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW,
                       ActiMode activation,
                       int groups,
                       bool use_bias,
                       const Layer* shared_op,
                       Initializer* kernel_initializer,
                       Initializer* bias_initializer,
                       char const *name)
{
  assert(input->num_dims == 4); /*NCHW*/

  Layer *conv = new Layer(this, OP_CONV2D, name, 1/*inputs*/,
                          use_bias ? 2 : 1/*weights*/, 1/*outputs*/,
                          input);
  {
    int numdims = 4;
    int dims[MAX_TENSOR_DIM];
    dims[3] = input->dims[3];
    dims[2] = outChannels;
    dims[1] = 1 + (input->dims[1] + 2 * paddingH - kernelH) / strideH;
    dims[0] = 1 + (input->dims[0] + 2 * paddingW - kernelW) / strideW;
    conv->outputs[0] = create_tensor_legion_ordering(numdims, dims, DT_FLOAT,
                                                     conv, 0, true/*create_grad*/);
  }
  {
    int dims[4] = {kernelW, kernelH, input->dims[2], outChannels};
    conv->weights[0] = create_weight_legion_ordering(4, dims, DT_FLOAT,
        conv, true/*create_grad*/, kernel_initializer, CHOSEN_SYNC_TYPE);
  }
  if (use_bias)
  {
    int dims[1] = {outChannels};
    conv->weights[1] = create_weight_legion_ordering(1, dims, DT_FLOAT,
        conv, true/*create_grad*/, bias_initializer, CHOSEN_SYNC_TYPE);
  }
  conv->add_int_property("out_channels", outChannels);
  conv->add_int_property("kernel_h", kernelH);
  conv->add_int_property("kernel_w", kernelW);
  conv->add_int_property("stride_h", strideH);
  conv->add_int_property("stride_w", strideW);
  conv->add_int_property("padding_h", paddingH);
  conv->add_int_property("padding_w", paddingW);
  conv->add_int_property("activation", activation);
  conv->add_int_property("groups", groups);
  conv->add_int_property("use_bias", use_bias);
  conv->add_initializer("kernel", kernel_initializer);
  conv->add_initializer("bias", bias_initializer);
  layers.push_back(conv);
  return conv->outputs[0];
}

Op* Conv2D::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  long long value;
  layer->get_int_property("out_channels", value);
  int out_channels = value;
  layer->get_int_property("kernel_h", value);
  int kernelH = value;
  layer->get_int_property("kernel_w", value);
  int kernelW = value;
  layer->get_int_property("stride_h", value);
  int strideH = value;
  layer->get_int_property("stride_w", value);
  int strideW = value;
  layer->get_int_property("padding_h", value);
  int paddingH = value;
  layer->get_int_property("padding_w", value);
  int paddingW = value;
  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode) value;
  layer->get_int_property("groups", value);
  int groups = value;
  layer->get_int_property("use_bias", value);
  bool use_bias = value;
  Initializer *kernel_initializer, *bias_initializer;
  layer->get_initializer("kernel", kernel_initializer);
  layer->get_initializer("bias", bias_initializer);
  return new Conv2D(model, 
      layer->layer_guid,
      inputs[0],
      out_channels,
      kernelH, kernelW,
      strideH, strideW, 
      paddingH, paddingW, 
      activation,
      groups,
      use_bias, 
      false, // allocate_weights
      layer->name);
}

Conv2DParams Conv2D::get_params() const {
  Conv2DParams params;
  params.layer_guid = this->layer_guid;
  params.out_channels = this->out_channels;
  params.kernel_h = this->kernel_h;
  params.kernel_w = this->kernel_w;
  params.stride_h = this->stride_h;
  params.stride_w = this->stride_w;
  params.padding_h = this->padding_h;
  params.padding_w = this->padding_w;
  params.activation = this->activation;
  params.groups = this->groups;
  params.use_bias = this->use_bias;

  return params;
}

size_t Conv2DParams::get_hash(const ParallelTensor input) const {
  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, this->layer_guid.id);
  hash_combine(hash, this->out_channels);
  hash_combine(hash, this->kernel_h);
  hash_combine(hash, this->kernel_w);
  hash_combine(hash, this->stride_h);
  hash_combine(hash, this->stride_w);
  hash_combine(hash, this->padding_h);
  hash_combine(hash, this->padding_w);
  hash_combine(hash, this->activation);
  hash_combine(hash, this->groups);
  hash_combine(hash, this->use_bias);

  return hash;
}

size_t Conv2D::get_params_hash() const {
  return this->get_params().get_hash(this->inputs[0]);
}

using PCG::Node;
Node FFModel::get_or_create_conv2d_node(const ParallelTensor input,
                                        const Conv2DParams& params)
{
  if (!params.is_valid(input)) {
    return Node::INVALID_NODE;
  }
  // Currently disable parallelizing the height and width dimension
  if (input->dims[0].degree > 1 || input->dims[1].degree > 1) {
    return Node::INVALID_NODE;
  }

  size_t hash = params.get_hash(input);

  Conv2D *conv = NULL;

  std::pair<ParallelTensorShape, Conv2DParams> key{input->get_shape(), params};
  const auto &it = this->cached_conv2d_ops.find(key);
  if (it != cached_conv2d_ops.end()) {
    conv = it->second;
  } else {
    conv = new Conv2D(*this,
                      params.layer_guid,
                      input, 
                      params.out_channels, 
                      params.kernel_h, params.kernel_w, 
                      params.stride_h, params.stride_w,
                      params.padding_h, params.padding_w,
                      params.activation, 
                      params.groups, 
                      params.use_bias,
                      false/*allocate_weights*/,
                      NULL);
    cached_conv2d_ops[key] = conv;
  }

  // size_t test_hash1 = 4577313447550563550;
  // hash_combine(test_hash1, 1000098);
  // hash_combine(test_hash1, 384);

  // size_t test_hash2 = 4577313447550563550;
  // hash_combine(test_hash2, 1000101);
  // hash_combine(test_hash2, 448);

  // assert (test_hash1 != test_hash2);

  // Conv2DParams test_params1;
  // LayerID layer_id1(1000098);
  // test_params1.layer_guid = layer_id1;
  // test_params1.out_channels = 384;
  // test_params1.kernel_h = 1;
  // test_params1.kernel_w = 1;
  // test_params1.stride_h = 1;
  // test_params1.stride_w = 1;
  // test_params1.padding_h = 0;
  // test_params1.padding_w = 0;
  // test_params1.groups = 1;
  // test_params1.activation = AC_MODE_NONE;
  // test_params1.use_bias = true;

  // Conv2DParams test_params2;
  // LayerID layer_id2(1000101);
  // test_params2.layer_guid = layer_id2;
  // test_params2.out_channels = 448;
  // test_params2.kernel_h = 1;
  // test_params2.kernel_w = 1;
  // test_params2.stride_h = 1;
  // test_params2.stride_w = 1;
  // test_params2.padding_h = 0;
  // test_params2.padding_w = 0;
  // test_params2.groups = 1;
  // test_params2.activation = AC_MODE_NONE;
  // test_params2.use_bias = true;

  // assert (test_params1.get_hash(input) != test_params2.get_hash(input));

  // size_t thing = 4574693366283544798;
  // size_t thing2 = thing;
  // hash_combine(thing, (int)384);
  // hash_combine(thing2, (int)448);
  // assert (thing != thing2);
  // assert (conv->get_params().get_hash(input) == hash);

  assert (conv->get_params() == params);
  return this->new_node(conv);
}

bool operator==(Conv2DParams const &lhs, Conv2DParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && 
      lhs.kernel_h == rhs.kernel_h &&
      lhs.kernel_w == rhs.kernel_w && 
      lhs.stride_h == rhs.stride_h && 
      lhs.stride_w == rhs.stride_w && 
      lhs.padding_h == rhs.padding_h && 
      lhs.padding_w == rhs.padding_w &&
      lhs.groups == rhs.groups && 
      lhs.activation == rhs.activation && 
      lhs.use_bias == rhs.use_bias;
}

Node FFModel::get_or_create_conv2d_node(const LayerID& layer_guid,
                                        const ParallelTensor input,
                                        int outChannels,
                                        int kernelH, int kernelW,
                                        int strideH, int strideW,
                                        int paddingH, int paddingW,
                                        ActiMode activation,
                                        int groups,
                                        bool use_bias) 
{
  Conv2DParams params;
  params.layer_guid = layer_guid;
  params.out_channels = outChannels;
  params.kernel_h = kernelH;
  params.kernel_w = kernelW;
  params.stride_h = strideH;
  params.stride_w = strideW;
  params.padding_h = paddingH;
  params.padding_w = paddingW;
  params.activation = activation;
  params.groups = groups;
  params.use_bias = use_bias;

  return this->get_or_create_conv2d_node(input, params);
}

void Conv2DParams::mark_replica_dims(const ParallelTensor input,
                                     ParallelDim output_dims[MAX_TENSOR_DIM], 
                                     ParallelDim kernel_dims[MAX_TENSOR_DIM], 
                                     ParallelDim bias_dims[MAX_TENSOR_DIM]) const 
{
  if (output_dims != nullptr) {
    output_dims[Conv2DOutput::REPLICA].is_replica_dim = true;
  }
  if (kernel_dims != nullptr) {
    kernel_dims[Conv2DOutput::REPLICA].is_replica_dim = true;
  }
  if (bias_dims != nullptr) {
    bias_dims[Conv2DBias::REPLICA_1].is_replica_dim = true;
    bias_dims[Conv2DBias::REPLICA_2].is_replica_dim = true;
    bias_dims[Conv2DBias::REPLICA_3].is_replica_dim = true;
    bias_dims[Conv2DBias::REPLICA_4].is_replica_dim = true;
  }
}

int Conv2DParams::output_size(const ParallelTensor input,
                              ParallelDim output_dims[MAX_TENSOR_DIM]) const {
  int input_w = input->dims[Conv2DInput::WIDTH].size;
  int input_h = input->dims[Conv2DInput::HEIGHT].size;

  output_dims[Conv2DOutput::SAMPLE].size = input->dims[Conv2DInput::SAMPLE].size;
  output_dims[Conv2DOutput::CHANNEL].size = out_channels;
  output_dims[Conv2DOutput::HEIGHT].size = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  output_dims[Conv2DOutput::WIDTH].size = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;

  return input->num_dims;
};

int Conv2DParams::kernel_size(const ParallelTensor input,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM]) const {
  kernel_dims[Conv2DKernel::CHANNEL_OUT].size = this->out_channels;
  kernel_dims[Conv2DKernel::CHANNEL_IN].size = input->dims[Conv2DInput::CHANNEL].size / this->groups;
  kernel_dims[Conv2DKernel::HEIGHT].size = this->kernel_h * input->dims[Conv2DInput::HEIGHT].degree;
  kernel_dims[Conv2DKernel::WIDTH].size = this->kernel_w * input->dims[Conv2DInput::WIDTH].degree;

  return Conv2DKernel::NUMDIM;
}

int Conv2DParams::bias_size(const ParallelTensor input,
                            ParallelDim bias_dims[MAX_TENSOR_DIM]) const {
  bias_dims[Conv2DBias::CHANNEL].size = this->out_channels;

  return Conv2DBias::NUMDIM;
};

void Conv2DParams::solve_dims(const ParallelTensor input, 
                              ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,  
                              ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const 
{
  assert ((output_dims == nullptr) == (output_ndims == nullptr));
  assert ((kernel_dims == nullptr) == (kernel_ndims == nullptr));
  assert ((bias_dims == nullptr) == (bias_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  Conv2D::construct_mappings(mapping, this->use_bias);

  this->mark_replica_dims(input, output_dims, kernel_dims, bias_dims);

  std::vector<ParallelDim *> output_dim_sets;
  if (output_dims != nullptr) {
    output_dim_sets.push_back(output_dims);
  }

  std::vector<ParallelDim *> weight_dim_sets;
  if (kernel_dims != nullptr) {
    weight_dim_sets.push_back(kernel_dims);
  }
  if (bias_dims != nullptr && this->use_bias) {
    weight_dim_sets.push_back(bias_dims);
  }

  solve_parallel_dim_mappings(
      mapping, 
      {input->dims},
      weight_dim_sets,
      output_dim_sets
  );

  if (output_dims != nullptr) {
    *output_ndims = this->output_size(input, output_dims);
  }
  if (kernel_dims != nullptr) {
    *kernel_ndims = this->kernel_size(input, kernel_dims);
  }
  if (bias_dims != nullptr && this->use_bias) {
    *bias_ndims = this->bias_size(input, bias_dims);
  }
}

/*static*/
void Conv2D::construct_mappings(std::vector<ParallelDimMappingRecord>& out, bool use_bias) {
  Conv2D::construct_output_mappings(out);
  Conv2D::construct_weight_mappings(out, use_bias);
}

/*static*/
void Conv2D::construct_output_mappings(std::vector<ParallelDimMappingRecord>& out) {
  Op::construct_output_parallel_dims(
    out, 
    {
      {Conv2DInput::CHANNEL, MappingOperation::REPLICATE, Conv2DOutput::REPLICA},
      {Conv2DInput::SAMPLE, MappingOperation::PARTITION, Conv2DOutput::SAMPLE},
      {Conv2DInput::REPLICA, MappingOperation::PARTITION, Conv2DOutput::CHANNEL},
      {Conv2DInput::HEIGHT, MappingOperation::PARTITION, Conv2DOutput::HEIGHT},
      {Conv2DInput::WIDTH, MappingOperation::PARTITION, Conv2DOutput::WIDTH}
    }
  );
}

/*static*/
void Conv2D::construct_weight_mappings(std::vector<ParallelDimMappingRecord>& out, bool use_bias) {
  Op::construct_weight_parallel_dims(
    out,
    {
      {Conv2DInput::REPLICA, MappingOperation::PARTITION, Conv2DKernel::CHANNEL_OUT},
      {Conv2DInput::SAMPLE, MappingOperation::REPLICATE, Conv2DKernel::REPLICA},
      {Conv2DInput::CHANNEL, MappingOperation::PARTITION, Conv2DKernel::CHANNEL_IN}, 
      {Conv2DInput::HEIGHT, MappingOperation::REPLICATE, Conv2DKernel::HEIGHT}, // Kernel::{HEIGHT, WEIGHT} would both work here
      {Conv2DInput::WIDTH, MappingOperation::REPLICATE, Conv2DKernel::WIDTH}, // same as above
    }, 
    Conv2DInput::INDEX, Conv2DKernel::INDEX
  );

  if (use_bias) {
    Op::construct_weight_parallel_dims(
      out,
      {
        {Conv2DInput::REPLICA, Conv2DBias::REPLICA_1},
        {Conv2DInput::SAMPLE, Conv2DBias::REPLICA_2},
        {Conv2DInput::CHANNEL, Conv2DBias::CHANNEL},
        {Conv2DInput::HEIGHT, Conv2DBias::REPLICA_3},
        {Conv2DInput::WIDTH, Conv2DBias::REPLICA_4}
      }, 
      Conv2DInput::INDEX, Conv2DBias::INDEX
    );
  }
}

Conv2D::Conv2D(FFModel& model,
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
         other.name) 
{ }

bool Conv2DParams::is_valid(const ParallelTensor input) const {
  ParallelTensorShape output_shape, kernel_shape, bias_shape;
  this->solve_dims(input, 
                   output_shape.dims, &output_shape.num_dims,
                   kernel_shape.dims, &kernel_shape.num_dims,
                   bias_shape.dims, &bias_shape.num_dims);
  bool is_valid = true;
  is_valid &= input->check_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= kernel_shape.is_valid();
  if (use_bias) { 
    is_valid &= bias_shape.is_valid();
  }

  return is_valid;
}

Conv2D::Conv2D(FFModel& model,
               const LayerID& _layer_guid,
               const ParallelTensor input,
               int outChannels,
               int kernelH, int kernelW,
               int strideH, int strideW, 
               int paddingH, int paddingW,
               ActiMode activation,
               int groups,
               bool use_bias,
               bool allocate_weights,
               const char* name)
: Op(model, OP_CONV2D, name, 1/*inputs*/, use_bias ? 2 : 1/*weights*/, allocate_weights, 1/*outputs*/, input),
  in_channels(input->dims[Conv2DInput::CHANNEL].size/input->dims[Conv2DInput::CHANNEL].degree),
  out_channels(outChannels),
  kernel_h(kernelH), kernel_w(kernelW),
  stride_h(strideH), stride_w(strideW),
  padding_h(paddingH), padding_w(paddingW),
  activation(activation),
  groups(groups),
  use_bias(use_bias)
{
  // overwrite layer_guid
  layer_guid = _layer_guid;
  assert (input->num_dims == Conv2DInput::NUMDIM);
  assert (this->stride_h > 0);
  assert (this->stride_w > 0);

  ParallelDim output_dims[MAX_TENSOR_DIM],
              kernel_dims[MAX_TENSOR_DIM], 
              bias_dims[MAX_TENSOR_DIM];
  int output_ndims,
      kernel_ndims,
      bias_ndims;

  this->construct_mappings(
      *this->parallel_dims_mapping, this->use_bias);
  this->get_params().solve_dims(
      this->inputs[0],
      output_dims, &output_ndims,
      kernel_dims, &kernel_ndims,
      bias_dims, &bias_ndims);

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand()/*seed*/);

    weights[Conv2DKernel::INDEX] = model.create_parallel_weight_legion_ordering(
        kernel_ndims, kernel_dims, DT_FLOAT, NULL/*owner_op*/, true/*create_grad*/, kernel_initializer, CHOSEN_SYNC_TYPE);
    
    if (use_bias) {
      Initializer *bias_initializer = new ZeroInitializer();

      weights[Conv2DBias::INDEX] = model.create_parallel_weight_legion_ordering(
          bias_ndims, bias_dims, DT_FLOAT, NULL/*owner_op*/, true/*create_grad*/, bias_initializer, CHOSEN_SYNC_TYPE);
    }
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(output_ndims, output_dims, DT_FLOAT, this);

  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Conv2D::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(CONV2D_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(3, FID_DATA);
  //launcher.add_region_requirement(
  //    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
  //                      WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  //launcher.add_field(4, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0]: input
  regions[1]: output
  regions[2](I): filter
  regions[3](I): bias
  regions[4](O): filter_grad
  regions[5](O): input_grad
*/
OpMeta* Conv2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Conv2D* conv = (Conv2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorR<float, Conv2DInput::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Conv2DOutput::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, Conv2DKernel::NUMDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, Conv2DBias::NUMDIM> acc_bias(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Conv2DKernel::NUMDIM> acc_kernel_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  //TensorAccessorW<float, 4> acc_input_grad(
  //    regions[4], task->regions[4], FID_DATA, ctx, runtime,
  //    false/*readOutput*/);

  Conv2DMeta* m = new Conv2DMeta(handle);
  m->relu = conv->activation == AC_MODE_RELU;
  m->use_bias = conv->use_bias;
  m->profiling = conv->profiling;
  m->trainableInputs[0] = conv->trainableInputs[0];
  std::strcpy(m->op_name, conv->name);

  int input_w = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_h = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int input_c = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int input_n = acc_input.rect.hi[3] - acc_input.rect.lo[3] + 1;
  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int output_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  int output_n = acc_output.rect.hi[3] - acc_output.rect.lo[3] + 1;

  printf("init conv (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n, input_c, input_h, input_w);
  printf("init conv (output): n(%d) c(%d) h(%d) w(%d)\n",
          output_n, output_c, output_h, output_w);

  //printf("convDim: padding(%d %d) stride(%d %d)\n", conv->padding_h, conv->padding_w, conv->stride_h, conv->stride_w);
  int pad_h = ((output_h - 1) * conv->stride_h + conv->kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * conv->stride_w + conv->kernel_w - input_w + 1) / 2;
  if (pad_h != conv->padding_h)
    printf("Warning: changing conv_padding_h to satisfy output_h size\n");
  if (pad_w != conv->padding_w)
    printf("Warning: changing conv_padding_w to satisfy output_w size\n");

  Conv2D::init_kernel(conv, m, 
                      input_w, input_h, input_c, input_n,
                      output_w, output_h, output_c, output_n,
                      pad_h, pad_w,
                      acc_input.ptr, acc_output.ptr, acc_kernel.ptr, acc_kernel_grad.ptr);

  return m;
}

void Conv2D::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(CONV2D_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(
        RegionRequirement(weights[1]->region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): filter
  regions[3](I): bias
*/
void Conv2D::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  //Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  assert(regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  TensorAccessorR<float, Conv2DInput::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Conv2DOutput::NUMDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, Conv2DKernel::NUMDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  const float* acc_bias_ptr = NULL;
  if (m->use_bias) { 
    TensorAccessorR<float, Conv2DBias::NUMDIM> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    acc_bias_ptr = acc_bias.ptr;
  }

  Conv2D::forward_kernel_wrapper(m, acc_input.ptr, acc_output.ptr, acc_kernel.ptr, acc_bias_ptr);
}

void Conv2D::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(CONV2D_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  int rid = 0;
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(rid++, FID_DATA);
  // regions[1](I/O): input_grad
  if (trainableInputs[0]) {
    launcher.add_region_requirement(
        RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
  }
  // regions[2](I): output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(rid++, FID_DATA);
  // regions[3](I/O): output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(rid++, FID_DATA);
  // regions[4](I): filter
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(rid++, FID_DATA);
  // regions[5](I/O): filter_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(rid++, FID_DATA);
  if (use_bias) {
    // regions[6](I/O): bias_grad
    launcher.add_region_requirement(
        RegionRequirement(weights[1]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, weights[1]->region_grad));
    launcher.add_field(rid++, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
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
void Conv2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  //Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  assert(regions.size() == (5 + static_cast<size_t>(m->trainableInputs[0]) + static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() == (5 + static_cast<size_t>(m->trainableInputs[0]) + static_cast<size_t>(m->use_bias)));
  size_t rid = 0;
  TensorAccessorR<float, Conv2DInput::NUMDIM> acc_input(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid ++;
  float* acc_input_grad_ptr = NULL;
  if (m->trainableInputs[0]) {
    TensorAccessorW<float, Conv2DInput::NUMDIM> acc_input_grad(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    acc_input_grad_ptr = acc_input_grad.ptr;
    rid ++;
  }
  TensorAccessorR<float, Conv2DOutput::NUMDIM> acc_output(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid ++;
  TensorAccessorW<float, Conv2DOutput::NUMDIM> acc_output_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid ++;
  TensorAccessorR<float, Conv2DKernel::NUMDIM> acc_kernel(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid ++;
  TensorAccessorW<float, Conv2DKernel::NUMDIM> acc_kernel_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid ++;
  float* acc_bias_grad_ptr = NULL;
  if (m->use_bias) { 
    TensorAccessorW<float, Conv2DBias::NUMDIM> acc_bias_grad(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    acc_bias_grad_ptr = static_cast<float*>(acc_bias_grad.ptr);
    rid ++;
  }
  assert(rid == regions.size());

  Conv2D::backward_kernel_wrapper(m, acc_input.ptr, acc_input_grad_ptr,
                                  acc_output.ptr, acc_output_grad.ptr,
                                  acc_kernel.ptr, acc_kernel_grad.ptr,
                                  acc_bias_grad_ptr);
}

void Conv2D::print_layer(const FFModel& ff)
{
  printf("conv2d layer\n");
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
#if 0
  TaskLauncher launcher(CONV2D_PRINT_TASK_ID, TaskArgument(NULL, 0));
  launcher.add_region_requirement(
    RegionRequirement(kernel->region, READ_ONLY, EXCLUSIVE, kernel->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(bias->region, READ_ONLY, EXCLUSIVE, bias->region));
  launcher.add_field(1, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
#else
  RegionRequirement kernel_req(weights[0]->region, READ_WRITE, EXCLUSIVE, weights[0]->region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

/*
  RegionRequirement kernel_grad_req(kernel->region_grad, READ_WRITE, EXCLUSIVE, kernel->region_grad);
  kernel_grad_req.add_field(FID_DATA);
  InlineLauncher kernel_grad_launcher(kernel_grad_req);
  PhysicalRegion kernel_grad_region = runtime->map_region(ctx, kernel_grad_launcher);
  kernel_grad_region.wait_until_valid();
*/
  RegionRequirement bias_req(weights[1]->region, READ_WRITE, EXCLUSIVE, weights[1]->region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();
/*
  RegionRequirement bias_grad_req(bias->region_grad, READ_WRITE, EXCLUSIVE, bias->region_grad);
  bias_grad_req.add_field(FID_DATA);
  InlineLauncher bias_grad_launcher(bias_grad_req);
  PhysicalRegion bias_grad_region = runtime->map_region(ctx, bias_grad_launcher);
  bias_grad_region.wait_until_valid();
  */
  TensorAccessorW<float, Conv2DKernel::NUMDIM> acc_kernel(kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
//  const AccessorRW<float, 1> acc_kernel_grad(kernel_grad_region, FID_DATA);
  TensorAccessorW<float, Conv2DBias::NUMDIM> acc_bias(bias_region, bias_req, FID_DATA, ctx, runtime, true);
  //const AccessorRW<float, 1> acc_bias_grad(bias_grad_region, FID_DATA);

  const float *kernel_ptr = acc_kernel.ptr;
  //float *kernel_grad_ptr = acc_kernel_grad.ptr;
  const float *bias_ptr = acc_bias.ptr;
  //float *bias_grad_ptr = acc_bias_grad.ptr;

  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  int kernel_dim3 = acc_kernel.rect.hi[2] - acc_kernel.rect.lo[2] + 1;
  int kernel_dim4 = acc_kernel.rect.hi[3] - acc_kernel.rect.lo[3] + 1;
  //size_t kernel_grad_size = rect_kernel_grad.volume();
  size_t bias_size = acc_bias.rect.volume();
  //size_t bias_grad_size = rect_bias_grad.volume();
  printf("kernel, %p, %zu, [%d, %d, %d, %d]\n", kernel_ptr, kernel_size, kernel_dim1, kernel_dim2, kernel_dim3, kernel_dim4);
  //printf("kernel_grad, %d\n", kernel_grad_size);
  printf("bias, %p, %zu\n", bias_ptr, bias_size);
  //printf("bias_grad, %d\n", bias_grad_size);


  for (size_t i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");

/*
  for (int i = 0; i < bias_grad_size; i++) {
    printf("%f ", bias_grad_ptr);
    bias_grad_ptr ++;
  }
  printf("\n");*/

  for (size_t i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");

/*
  for (int i = 0; i < kernel_grad_size; i++) {
    printf("%f ", kernel_grad_ptr);
    kernel_grad_ptr ++;
  }
  printf("\n");
  */
  runtime->unmap_region(ctx, kernel_region);
 // runtime->unmap_region(ctx, kernel_grad_region);
  runtime->unmap_region(ctx, bias_region);
//  runtime->unmap_region(ctx, bias_grad_region);
#endif
}

bool Conv2D::estimate_sync_cost(Simulator* sim, 
                                const MachineView& view,
                                CostMetrics& cost_metrics) const 
{
  ParallelDim kernel_dims[MAX_TENSOR_DIM],
              bias_dims[MAX_TENSOR_DIM];
  int kernel_ndims,
      bias_ndims;
  
  this->get_params().solve_dims(this->inputs[0], 
                                nullptr, nullptr,
                                kernel_dims, &kernel_ndims,
                                bias_dims, &bias_ndims);

  cost_metrics.sync_time = sim->default_estimate_sync_cost(kernel_dims, kernel_ndims, view);

  if (this->use_bias) {
    cost_metrics.sync_time += sim->default_estimate_sync_cost(bias_dims, bias_ndims, view);
  }

  return true;
}

void Conv2D::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->out_channels);
  sez.serialize(this->kernel_h);
  sez.serialize(this->kernel_w);
  sez.serialize(this->stride_h);
  sez.serialize(this->stride_w);
  sez.serialize(this->padding_h);
  sez.serialize(this->padding_w);
  sez.serialize(this->groups);
  sez.serialize(this->use_bias);
  sez.serialize(this->activation);
}

using PCG::Node;
/*static*/
Node Conv2D::deserialize(FFModel& ff, Legion::Deserializer& dez, ParallelTensor inputs[], int num_inputs) {
  assert (num_inputs == 1);

  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  bool use_bias;
  ActiMode activation;
  size_t id;
  dez.deserialize(id);
  LayerID layer_guid(id);
  dez.deserialize(out_channels);
  dez.deserialize(kernel_h);
  dez.deserialize(kernel_w);
  dez.deserialize(stride_h);
  dez.deserialize(stride_w);
  dez.deserialize(padding_h);
  dez.deserialize(padding_w);
  dez.deserialize(groups);
  dez.deserialize(use_bias);
  dez.deserialize(activation);

  return ff.get_or_create_conv2d_node(
      layer_guid,
      inputs[0],
      out_channels,
      kernel_h, kernel_w,
      stride_h, stride_w,
      padding_h, padding_w,
      activation,
      groups,
      use_bias);
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

}; // namespace FlexFlow

namespace std {
  size_t hash<FlexFlow::Conv2DParams>::operator()(FlexFlow::Conv2DParams const &params) const {
    size_t key = 0;
    hash_combine(key, params.layer_guid.id);
    hash_combine(key, params.out_channels);
    hash_combine(key, params.kernel_h);
    hash_combine(key, params.kernel_w);
    hash_combine(key, params.stride_h);
    hash_combine(key, params.stride_w);
    hash_combine(key, params.padding_h);
    hash_combine(key, params.padding_w);
    hash_combine(key, params.activation);
    hash_combine(key, params.groups);
    hash_combine(key, params.use_bias);
    return key;
  }
};
