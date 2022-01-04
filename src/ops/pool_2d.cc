#include "flexflow/ops/pool_2d.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::InlineLauncher;

Tensor FFModel::pool2d(const Tensor input,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW,
                       PoolType type, ActiMode activation,
                       char const *name)
{
  assert(input->num_dims == 4); /*NCHW*/
  Layer *pool = new Layer(this, OP_POOL2D, name, 1/*inputs*/,
                          0/*weights*/, 1/*outputs*/, input);
  int numdims = 4;
  int dims[MAX_TENSOR_DIM];
  dims[3] = input->dims[3];
  dims[2] = input->dims[2];
  dims[1] = 1 + (input->dims[1] + 2 * paddingH - kernelH) / strideH;
  dims[0] = 1 + (input->dims[0] + 2 * paddingW - kernelW) / strideW;
  pool->outputs[0] = create_tensor_legion_ordering(numdims, dims, DT_FLOAT,
                                                   pool, 0, true/*create_grad*/);
  pool->add_int_property("kernel_h", kernelH);
  pool->add_int_property("kernel_w", kernelW);
  pool->add_int_property("stride_h", strideH);
  pool->add_int_property("stride_w", strideW);
  pool->add_int_property("padding_h", paddingH);
  pool->add_int_property("padding_w", paddingW);
  pool->add_int_property("pool_type", type);
  pool->add_int_property("activation", activation);
  layers.push_back(pool);
  return pool->outputs[0];

#ifdef DEACODE
  Pool2D *pool = new Pool2D(*this, input, kernelH, kernelW,
                      strideH, strideW, paddingH, paddingW,
                      type, activation, name);
  layers.push_back(pool);
  return pool->outputs[0];
#endif
}

Op* Pool2D::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  long long value;
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
  layer->get_int_property("pool_type", value);
  PoolType type = (PoolType) value;
  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode) value;
  return new Pool2D(model, inputs[0], kernelH, kernelW,
      strideH, strideW, paddingH, paddingW, type, activation, layer->name);
}

Pool2DParams Pool2D::get_params() const {
  Pool2DParams params;
  params.kernel_h = this->kernel_h;
  params.kernel_w = this->kernel_w;
  params.stride_h = this->stride_h;
  params.stride_w = this->stride_w;
  params.padding_h = this->padding_h;
  params.padding_w = this->padding_w;
  params.pool_type = this->pool_type;
  params.activation = this->activation;

  return params;
}

bool Pool2DParams::is_valid(const ParallelTensor input) const {
  ParallelTensorShape output_shape;

  this->solve_dims(
      input, 
      output_shape.dims, &output_shape.num_dims
  );

  bool is_valid = true;
  is_valid &= input->check_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= (input->dims[Pool2DInput::REPLICA].degree == 1);

  return is_valid;
}

size_t Pool2DParams::get_hash(const ParallelTensor input) const {
  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, this->kernel_h);
  hash_combine(hash, this->kernel_w);
  hash_combine(hash, this->stride_h);
  hash_combine(hash, this->stride_w);
  hash_combine(hash, this->padding_h);
  hash_combine(hash, this->padding_w);
  hash_combine(hash, this->pool_type);
  hash_combine(hash, this->activation);

  return hash;
}

size_t Pool2D::get_params_hash() const {
  return this->get_params().get_hash(this->inputs[0]);
}

using PCG::Node;
Node FFModel::get_or_create_pool2d_node(const ParallelTensor input,
                                        const Pool2DParams& params)
{
  if (!params.is_valid(input)) {
    return Node::INVALID_NODE;
  }
  // Currently disable parallelizing the height and width dimension
  if (input->dims[0].degree > 1 || input->dims[1].degree > 1) {
    return Node::INVALID_NODE;
  }

  Pool2D *pool;

  size_t hash = params.get_hash(input);

  const auto &it = this->cached_pool2d_ops.find(hash);
  if (it != cached_pool2d_ops.end()) {
    pool = it->second;
  } else {
    pool = new Pool2D(*this, 
                      input, 
                      params.kernel_h, params.kernel_w, 
                      params.stride_h, params.stride_w,
                      params.padding_h, params.padding_w,
                      params.pool_type,
                      params.activation, 
                      NULL);
    cached_pool2d_ops[hash] = pool;
  }

  return this->new_node(pool);
}

Node FFModel::get_or_create_pool2d_node(const ParallelTensor input,
                                        int kernelH, int kernelW,
                                        int strideH, int strideW,
                                        int paddingH, int paddingW,
                                        PoolType type,
                                        ActiMode activation) 
{
  Pool2DParams params;
  params.kernel_h = kernelH;
  params.kernel_w = kernelW;
  params.stride_h = strideH;
  params.stride_w = strideW;
  params.padding_h = paddingH;
  params.padding_w = paddingW;
  params.pool_type = type;
  params.activation = activation;

  return this->get_or_create_pool2d_node(input, params);
}

int Pool2DParams::output_size(const ParallelTensor input, ParallelDim output_dims[MAX_TENSOR_DIM]) const { 
  int input_w = input->dims[Pool2DInput::WIDTH].size;
  int input_h = input->dims[Pool2DInput::HEIGHT].size;
  int input_c = input->dims[Pool2DInput::CHANNEL].size;
  int input_n = input->dims[Pool2DInput::SAMPLE].size;

  output_dims[Pool2DOutput::WIDTH].size = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  output_dims[Pool2DOutput::HEIGHT].size = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  output_dims[Pool2DOutput::CHANNEL].size = input_c;
  output_dims[Pool2DOutput::SAMPLE].size = input_n;
  output_dims[Pool2DOutput::REPLICA].is_replica_dim = true;

  return Pool2DOutput::NUMDIM;
}

void Pool2DParams::solve_dims(const ParallelTensor input, 
                ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims) const 
{
  assert ((output_dims == nullptr) == (output_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  Pool2D::construct_output_mappings(mapping);

  std::vector<ParallelDim *> output_dim_sets;
  if (output_dims != nullptr) {
    *output_ndims = this->output_size(input, output_dims);
    output_dim_sets.push_back(output_dims);
  }

  solve_parallel_dim_mappings(
      mapping,
      {input->dims},
      {},
      output_dim_sets
  );
}

/*static*/
void Pool2D::construct_output_mappings(std::vector<ParallelDimMappingRecord>& mappings) {
  Op::construct_output_parallel_dims(
    mappings,
    {
      {Pool2DInput::REPLICA, MappingOperation::PARTITION, Pool2DOutput::REPLICA},
      {Pool2DInput::SAMPLE, MappingOperation::PARTITION, Pool2DOutput::SAMPLE},
      {Pool2DInput::CHANNEL, MappingOperation::PARTITION, Pool2DOutput::CHANNEL},
      {Pool2DInput::HEIGHT, MappingOperation::PARTITION, Pool2DOutput::HEIGHT},
      {Pool2DInput::WIDTH, MappingOperation::PARTITION, Pool2DOutput::WIDTH},
    }
  );
}

Pool2D::Pool2D(FFModel& model,
               Pool2D const &other,
               ParallelTensor const input) 
: Pool2D(model,
         input,
         other.kernel_h,
         other.kernel_w,
         other.stride_h,
         other.stride_w,
         other.padding_h,
         other.padding_w,
         other.pool_type,
         other.activation,
         other.name) 
{ }

Pool2D::Pool2D(FFModel& model,
               const ParallelTensor _input,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               PoolType _type, ActiMode _activation,
               const char* name)
: Op(model, OP_POOL2D, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, _input),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  pool_type(_type), activation(_activation)
{
  assert (_input->num_dims == Pool2DInput::NUMDIM);

  Pool2D::construct_output_mappings(*this->parallel_dims_mapping);

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndims;
  this->get_params().solve_dims(
      this->inputs[0],
      output_dims,
      &output_ndims
  );

  outputs[0] = model.create_parallel_tensor_legion_ordering(output_ndims, output_dims, DT_FLOAT, this);
}

void Pool2D::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(POOL2D_INIT_TASK_ID, parallel_is,
                              TaskArgument(this, sizeof(Pool2D)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0]->region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Pool2D::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(POOL2D_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Pool2D::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(POOL2D_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I): output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Pool2D::serialize(Legion::Serializer& sez) const {
  sez.serialize(this->kernel_h);
  sez.serialize(this->kernel_w);
  sez.serialize(this->stride_h);
  sez.serialize(this->stride_w);
  sez.serialize(this->padding_h);
  sez.serialize(this->padding_w);
  sez.serialize(this->pool_type);
  sez.serialize(this->activation);
}

using PCG::Node;
/*static*/
Node Pool2D::deserialize(FFModel& ff, Legion::Deserializer& dez, ParallelTensor inputs[], int num_inputs) { 
  assert (num_inputs == 1);

  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;

  dez.deserialize(kernel_h);
  dez.deserialize(kernel_w);
  dez.deserialize(stride_h);
  dez.deserialize(stride_w);
  dez.deserialize(padding_h);
  dez.deserialize(padding_w);
  dez.deserialize(pool_type);
  dez.deserialize(activation);

  return ff.get_or_create_pool2d_node(
      inputs[0],
      kernel_h, kernel_w,
      stride_h, stride_w,
      padding_h, padding_w,
      pool_type,
      activation);
}

}; // namespace FlexFlow
