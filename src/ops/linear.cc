#include "ops/linear.h"
#include "legion/legion_utilities.h"
#include "hash_utils.h"

static constexpr int KERNEL_IDX = 0;
static constexpr int BIAS_IDX = 1;

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
  in_channels(_input->dims[0].size),
  out_channels(out_dim),
  activation(_activation),
  use_bias(_use_bias)
{
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
  outputs[0] = model.create_tensor_legion_ordering(output_shape.num_dims, output_shape.dims, DT_FLOAT);

  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

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
