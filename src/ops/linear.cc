#include "flexflow/ops/linear.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::InlineLauncher;

static constexpr int KERNEL_IDX = 0;
static constexpr int BIAS_IDX = 1;

Tensor FFModel::dense(const Tensor input,
                      int outDim,
                      ActiMode activation,
                      bool use_bias,
                      const Op* shared_op,
                      Initializer* kernel_initializer,
                      Initializer* bias_initializer,
                      const char *name)
{
  Linear* li = new Linear(*this, input, outDim, activation, use_bias, false, name);
  layers.push_back(li);
  return li->outputs[0];
}

size_t Linear::get_params_hash() const {
  return this->get_params().get_hash(this->inputs[0]);
}

Linear::Linear(FFModel& model,
               Linear const &other, 
               const Tensor input,
               bool allocate_weights)
: Linear(model, input, other.out_channels, other.activation, other.use_bias, allocate_weights, other.name)
{ }

Linear::Linear(FFModel& model,
               const Tensor _input,
               int out_dim,
               ActiMode _activation,
               bool _use_bias,
               bool allocate_weights,
               const char* name)
: Op(
    model, 
    OP_LINEAR, 
    name, 
    1/*inputs*/, 
    _use_bias ? 2 : 1 /*weights*/, 
    allocate_weights,
    1/*outputs*/, 
    _input),
  out_channels(out_dim),
  activation(_activation),
  use_bias(_use_bias)
{
  auto dimension_names = this->get_params().get_dimension_names(_input->get_shape());
  this->in_channels = _input->dims[dimension_names.at(LinearParams::INPUT_CHANNEL)].size;
    
  TensorShape input_shape = this->inputs[0]->get_shape();
  TensorShape output_shape, kernel_shape, bias_shape;
  LinearParams params = this->get_params();
  params.construct_mappings(*this->parallel_dims_mapping, input_shape);
  params.solve_dims(input_shape, output_shape, kernel_shape, bias_shape);

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand()/*seed*/);

    weights[KERNEL_IDX] = model.create_weight_legion_ordering(kernel_shape.num_dims, 
                                                              kernel_shape.dims, 
                                                              DT_FLOAT, 
                                                              NULL/*owner_op*/,
                                                              true/*create_grad*/,
                                                              kernel_initializer,
                                                              CHOSEN_SYNC_TYPE);

    if (use_bias) {
      Initializer *bias_initializer = new ZeroInitializer();

      weights[BIAS_IDX] = model.create_weight_legion_ordering(bias_shape.num_dims,
                                                              bias_shape.dims,
                                                              DT_FLOAT,
                                                              NULL/*owner_op*/,
                                                              true/*create_grad*/,
                                                              bias_initializer,
                                                              CHOSEN_SYNC_TYPE);
    }
  }

  // Create the output tensor
  outputs[0] = model.create_tensor_legion_ordering(output_shape.num_dims, output_shape.dims, DT_FLOAT, this);

  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Linear::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  //assert(check_output_input_weight_same_machine_view());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LINEAR_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  //launcher.add_region_requirement(
  //    RegionRequirement(input_lps[0], 0/*projection id*/,
  //                      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  //launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(1, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    //launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    //launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Linear::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(LINEAR_FWD_TASK_ID, parallel_is,
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
        RegionRequirement(weights[1]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Linear::backward(const FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  {
    ArgumentMap argmap;
    set_argumentmap_for_backward(ff, argmap);
    IndexLauncher launcher(LINEAR_BWD_TASK_ID, parallel_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           outputs[0]->machine_view.hash());
    int rid = 0;
    // regions[0](I): input
    launcher.add_region_requirement(
        RegionRequirement(inputs[0]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[1](I/O): replica_grad
    assert(replica == NULL);
    launcher.add_region_requirement(
        RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
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
    runtime->execute_index_space(ctx, launcher);
  }
  assert(replica == NULL);
}

void Linear::print_layer(const FFModel& ff)
{
  printf("linear layer\n");
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  RegionRequirement kernel_req(weights[0]->region, READ_WRITE, EXCLUSIVE, weights[0]->region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

  RegionRequirement bias_req(weights[1]->region, READ_WRITE, EXCLUSIVE, weights[1]->region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();

  TensorAccessorW<float, 2> acc_kernel(kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
  TensorAccessorW<float, 1> acc_bias(bias_region, bias_req, FID_DATA, ctx, runtime, true);

  const float *kernel_ptr = acc_kernel.ptr;
  const float *bias_ptr = acc_bias.ptr;

  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  size_t bias_size = acc_bias.rect.volume();
  printf("kernel, %p, %zu, [%d, %d]\n", kernel_ptr, kernel_size, kernel_dim1, kernel_dim2);
  printf("bias, %p, %zu\n", bias_ptr, bias_size);

  for (size_t i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");

  for (size_t i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");

  runtime->unmap_region(ctx, kernel_region);
  runtime->unmap_region(ctx, bias_region);

}

bool Linear::estimate_sync_cost(Simulator* sim,
                                const MachineView& view,
                                CostMetrics& cost_metrics) const
{
  // Estimate the cost of sync weights
  TensorShape tensor_shape;
  tensor_shape.num_dims = 3;
  tensor_shape.data_type = DT_FLOAT;
  tensor_shape.dims[0] = inputs[0]->dims[0];
  tensor_shape.dims[1] = inputs[0]->dims[inputs[0]->num_dims-1];
  tensor_shape.dims[2] = inputs[0]->dims[inputs[0]->num_dims-2];
  tensor_shape.dims[1].size = out_channels;
  tensor_shape.dims[1].degree = 1;
  tensor_shape.dims[2].degree = inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  tensor_shape.dims[2].size = inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  cost_metrics.sync_time = sim->default_estimate_sync_cost(tensor_shape, view, 1);
  //printf("[Estimate Linear] name(%s) sync_time(%.4lf)\n", name, cost_metrics.sync_time);
  return true;
}

ParallelConfig Linear::get_random_parallel_config(const FFModel& ff) const
{
  if (!ff.config.enable_parameter_parallel)
    return Op::get_random_parallel_config(ff);
  std::vector<int> batch_candidates;
  std::vector<int> channel_candidates;
  int batch = outputs[0]->dims[outputs[0]->num_dims-1].size;
  int channel = outputs[0]->dims[0].size;
  int total_devices = ff.config.workersPerNode * ff.config.numNodes;
  for (int i = 1; i <= ff.config.workersPerNode; i++)
    if (channel % i == 0)
      for (int j = 1; i * j <= total_devices; j++)
        if (batch % j == 0) {
          batch_candidates.push_back(j);
          channel_candidates.push_back(i);
        }
  assert(batch_candidates.size() > 0);
  int idx = std::rand() % batch_candidates.size();
  int num_par_c = channel_candidates[idx];
  int num_par_b = batch_candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  pc.dim[0] = num_par_c;
  pc.dim[pc.nDims-1] = num_par_b;
  for (int i = 1; i < pc.nDims - 1; i++)
    pc.dim[i] = 1;
  int start_idx = std::rand() % (total_devices - num_par_c * num_par_b + 1);
  start_idx = start_idx - start_idx % num_par_c;
  for (int i = 0; i < num_par_c * num_par_b; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

bool Linear::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_ACTI:
      *value = (int) activation;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Linear::is_valid_parallel_config(const FFModel& ff, const ParallelConfig& pc) const
{
  if (!ff.config.enable_parameter_parallel)
    return Op::is_valid_parallel_config(ff, pc);
  // Support data and parameter parallel
  if (pc.nDims != outputs[0]->num_dims)
    return false;
  for (int i = 1; i < pc.nDims-1; i++)
    if (pc.dim[i] != 1)
      return false;
  return true;
}

bool Linear::use_activation(ActiMode mode)
{
  switch (mode) {
    case AC_MODE_RELU:
    case AC_MODE_SIGMOID:
    case AC_MODE_TANH:
      return true;
  }
  return false;
}

using PCG::Node;
Node FFModel::get_or_create_linear_node(const Tensor input,
                                        const LinearParams& params) 
{
  if (!params.is_valid(input->get_shape())) {
    return Node::INVALID_NODE;
  }

  size_t hash = params.get_hash(input);
  
  Linear* li = NULL;
  auto it = cached_linear_ops.find(hash);
  if (it != cached_linear_ops.end()) {
    li = it->second;
  } else {
    li = new Linear(*this, input, params.out_channels, params.activation, params.use_bias, false/*allocate_weights*/, NULL);
    cached_linear_ops[hash] = li;
  }

  return this->new_node(li);
}

Node FFModel::get_or_create_linear_node(const Tensor input,
                                        int out_dim,
                                        ActiMode activation,
                                        bool use_bias)
{
  LinearParams params;
  params.out_channels = out_dim;
  params.activation = activation;
  params.use_bias = use_bias;

  auto dimension_names = params.get_dimension_names(input->get_shape()); 
  params.in_channels = input->dims[dimension_names.at(LinearParams::INPUT_CHANNEL)].size;

  return this->get_or_create_linear_node(input, params);
}

void Linear::serialize(Legion::Serializer& sez) const { 
  sez.serialize(this->out_channels); 
  sez.serialize(this->activation); 
  sez.serialize(this->use_bias); 
} 

/* static */
Node Linear::deserialize(FFModel &ff, Legion::Deserializer &dez, Tensor inputs[], int num_inputs) { 
  assert (num_inputs == 1); 
  int out_channels; 
  ActiMode activation; 
  bool use_bias; 
  dez.deserialize(out_channels); 
  dez.deserialize(activation); 
  dez.deserialize(use_bias); 
  return ff.get_or_create_linear_node(inputs[0], out_channels, activation, use_bias); 
} 

LinearParams Linear::get_params() const {
  LinearParams params;
  params.out_channels = this->out_channels;
  params.use_bias = this->use_bias;
  params.activation = this->activation;
  params.in_channels = this->in_channels;

  return params;
}

size_t LinearParams::get_hash(const Tensor input) const {
  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, this->out_channels);
  hash_combine(hash, this->activation);
  hash_combine(hash, this->use_bias);
  // hash_combine(hash, this->in_channels); // TODO FIXME @lockshaw do we need in_channels in the hash (and in params)
  return hash;
}

bool LinearParams::is_valid(TensorShape const &input_shape) const {
  TensorShape output_shape, kernel_shape, bias_shape;
  this->solve_dims(input_shape,
                   output_shape.dims, &output_shape.num_dims,
                   kernel_shape.dims, &kernel_shape.num_dims,
                   bias_shape.dims, &bias_shape.num_dims);
  bool is_valid = true;
  is_valid &= input_shape.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= kernel_shape.is_valid();
  if (use_bias) {
    is_valid &= bias_shape.is_valid();
  }
  return is_valid;
}

void LinearParams::solve_dims(const Tensor input,
                              ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const {
  this->solve_dims(input->get_shape(), 
                   output_dims, output_ndims,
                   kernel_dims, kernel_ndims,
                   bias_dims, bias_ndims);
}

void LinearParams::solve_dims(TensorShape const &input_shape,
                              TensorShape& output_shape,
                              TensorShape& kernel_shape,
                              TensorShape& bias_shape) const {
  this->solve_dims(input_shape, 
                   output_shape.dims, &output_shape.num_dims,
                   kernel_shape.dims, &kernel_shape.num_dims,
                   bias_shape.dims, &bias_shape.num_dims);
}

void LinearParams::solve_dims(TensorShape const &input_shape,
                              ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const {
  assert ((output_dims == nullptr) == (output_ndims == nullptr));
  assert ((kernel_dims == nullptr) == (kernel_ndims == nullptr));
  assert ((bias_dims == nullptr) == (bias_ndims == nullptr));
  {
    auto dimension_names = this->get_dimension_names(input_shape);
    assert (input_shape.dims[dimension_names.at(INPUT_CHANNEL)].size == this->in_channels);
  }

  std::vector<ParallelDimMappingRecord> mapping;
  this->construct_mappings(mapping, input_shape);
  this->mark_replica_dims(input_shape, output_dims, kernel_dims, bias_dims);

  solve_parallel_dim_mappings(mapping,
                              {input_shape.dims},
                              {kernel_dims, bias_dims},
                              {output_dims});

  this->calculate_nonreplica_dim_sizes(input_shape,
                                       output_dims, output_ndims,
                                       kernel_dims, kernel_ndims,
                                       bias_dims, bias_ndims);
}

std::unordered_map<LinearParams::NamedDimensions, int> LinearParams::get_dimension_names(TensorShape const &input_shape) const {
  int num_dims = input_shape.num_dims;

  return {
    {INPUT_CHANNEL, 0},
    {INPUT_SAMPLE, num_dims - 2},
    {INPUT_REPLICA, num_dims - 1},
    {OUTPUT_CHANNEL, 0},
    {OUTPUT_SAMPLE, num_dims - 2},
    {OUTPUT_REPLICA, num_dims - 1},
    {KERNEL_CHANNEL_IN, 0},
    {KERNEL_CHANNEL_OUT, 1},
    {BIAS_CHANNEL_OUT, 0}
  };
}

void LinearParams::calculate_nonreplica_dim_sizes(TensorShape const &input_shape,
                                                  ParallelDim output_dims[MAX_TENSOR_DIM], int* output_ndims,
                                                  ParallelDim kernel_dims[MAX_TENSOR_DIM], int* kernel_ndims,
                                                  ParallelDim bias_dims[MAX_TENSOR_DIM], int* bias_ndims) const {
  auto dimension_names = this->get_dimension_names(input_shape);
  int num_dims = input_shape.num_dims;

  if (output_dims != nullptr) {
    for (int i = 1; i < input_shape.num_dims - 1; i++) {
      output_dims[i].size = input_shape.dims[i].size;
    }
    output_dims[dimension_names.at(OUTPUT_CHANNEL)].size = this->out_channels;
    *output_ndims = num_dims;
  }
  if (kernel_dims != nullptr) {
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_IN)].size = this->in_channels;
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_OUT)].size = this->out_channels;
    *kernel_ndims = num_dims;
  }
  if (bias_dims != nullptr) {
    bias_dims[dimension_names.at(BIAS_CHANNEL_OUT)].size = this->out_channels;
    *bias_ndims = num_dims;
  }
}


void LinearParams::mark_replica_dims(TensorShape const &input_shape,
                                     ParallelDim output_dims[MAX_TENSOR_DIM], 
                                     ParallelDim kernel_dims[MAX_TENSOR_DIM],
                                     ParallelDim bias_dims[MAX_TENSOR_DIM]) const 
{
  int num_dims = input_shape.num_dims;
  auto dimension_names = this->get_dimension_names(input_shape);
  if (output_dims != nullptr) {
    output_dims[dimension_names.at(OUTPUT_REPLICA)].is_replica_dim = true;
  }
  if (kernel_dims != nullptr) {
    for (int i = 2; i < num_dims; i++) {
      kernel_dims[i].is_replica_dim = true;
    }
  }
  if (bias_dims != nullptr) {
    for (int i = 1; i < num_dims; i++) {
      bias_dims[i].is_replica_dim = true;
    }
  }
}

void LinearParams::construct_mappings(std::vector<ParallelDimMappingRecord>& mappings, TensorShape const &input_shape) const {
  std::unordered_map<NamedDimensions, int> dimension_names = this->get_dimension_names(input_shape);

  Op::construct_output_parallel_dims( 
    mappings,
    {
      {dimension_names.at(INPUT_CHANNEL), dimension_names.at(OUTPUT_REPLICA)},
      {dimension_names.at(INPUT_REPLICA), dimension_names.at(OUTPUT_CHANNEL)}
    }
  );
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_output_parallel_dims(mappings, i, i);
  }

  Op::construct_weight_parallel_dims(
    mappings,
    {
      {dimension_names.at(INPUT_CHANNEL), dimension_names.at(KERNEL_CHANNEL_IN)},
      {dimension_names.at(INPUT_REPLICA), dimension_names.at(KERNEL_CHANNEL_OUT)}
    }, 0/*input_idx*/, KERNEL_IDX
  );
  // map a bunch of replica dimensions for the unnamed dimensions in the input
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(mappings, i, i+1, 0/*input_idx*/, KERNEL_IDX); 
  }

  Op::construct_weight_parallel_dims(
    mappings,
    {
      {dimension_names.at(INPUT_REPLICA), dimension_names.at(BIAS_CHANNEL_OUT)},
    }, 0/*input_idx*/, BIAS_IDX
  );
  for (int i = 1; i < input_shape.num_dims; i++) {
    Op::construct_weight_parallel_dims(mappings, i, i+1, 0/*input_idx*/, BIAS_IDX);
  }
}

}; // namespace FlexFlow
