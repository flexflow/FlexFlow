#include "flexflow/operator.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#include "flexflow/simulator.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif
#include <stdexcept>

using namespace Legion;

namespace FlexFlow {

ParallelDimMappingRecord::ParallelDimMappingRecord(MappingRecordType type)
    : type(type), output_dim(-1), input_dim(-1), weight_dim(-1), output_idx(-1),
      input_idx(-1), weight_idx(-1) {}

/*static*/
ParallelDimMappingRecord ParallelDimMappingRecord::input_output_record(
    int input_idx,
    int input_dim,
    int output_idx,
    int output_dim,
    tl::optional<MappingOperation> operation) {
  ParallelDimMappingRecord r(MappingRecordType::INPUT_OUTPUT);
  r.operation = operation;

  assert(output_idx >= 0);
  assert(output_dim >= 0);
  assert(input_idx >= 0);
  assert(input_dim >= 0);

  r.output_idx = output_idx;
  r.output_dim = output_dim;
  r.input_idx = input_idx;
  r.input_dim = input_dim;

  return r;
}

/*static*/
ParallelDimMappingRecord ParallelDimMappingRecord::input_weight_record(
    int input_idx,
    int input_dim,
    int weight_idx,
    int weight_dim,
    tl::optional<MappingOperation> operation) {
  ParallelDimMappingRecord r(MappingRecordType::INPUT_WEIGHT);
  r.operation = operation;

  assert(input_idx >= 0);
  assert(input_dim >= 0);
  assert(weight_idx >= 0);
  assert(weight_dim >= 0);

  r.input_idx = input_idx;
  r.input_dim = input_dim;
  r.weight_idx = weight_idx;
  r.weight_dim = weight_dim;

  return r;
}

MappingRecordType ParallelDimMappingRecord::get_type() const {
  return this->type;
}

void Op::prefetch(FFModel const &ff) {
  // TODO: perform prefetch for performance imporvement
}

std::vector<ParallelTensor> Op::get_inputs() const {
  return {this->inputs, this->inputs + this->numInputs};
}

/*static*/
void Op::construct_weight_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx,
    int weight_idx) {
  for (std::tuple<int, MappingOperation, int> const &mapping : mappings) {
    Op::construct_weight_parallel_dims(records,
                                       std::get<0>(mapping),
                                       std::get<2>(mapping),
                                       input_idx,
                                       weight_idx,
                                       std::get<1>(mapping));
  }
}

/*static*/
void Op::construct_weight_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::pair<int, int>> mappings,
    int input_idx,
    int weight_idx) {
  for (std::pair<int, int> const &mapping : mappings) {
    Op::construct_weight_parallel_dims(
        records, mapping.first, mapping.second, input_idx, weight_idx);
  }
}

/*static*/
void Op::construct_weight_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    int input_dim,
    int weight_dim,
    int input_idx,
    int weight_idx,
    tl::optional<MappingOperation> operation) {
  records.push_back(ParallelDimMappingRecord::input_weight_record(
      input_idx, input_dim, weight_idx, weight_dim, operation));
}

void Op::register_weight_parallel_dims(
    std::vector<std::pair<int, int>> mappings, int input_idx, int weight_idx) {
  Op::construct_weight_parallel_dims(
      *this->parallel_dims_mapping, mappings, input_idx, weight_idx);
}

void Op::register_weight_parallel_dims(
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx,
    int weight_idx) {
  Op::construct_weight_parallel_dims(
      *this->parallel_dims_mapping, mappings, input_idx, weight_idx);
}

void Op::register_weight_parallel_dims(
    int input_dim,
    int weight_dim,
    int input_idx,
    int weight_idx,
    tl::optional<MappingOperation> operation) {
  Op::construct_weight_parallel_dims(*this->parallel_dims_mapping,
                                     input_dim,
                                     weight_dim,
                                     input_idx,
                                     weight_idx,
                                     operation);
}

/*static*/
void Op::construct_output_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx,
    int output_idx) {
  for (std::tuple<int, MappingOperation, int> const &mapping : mappings) {
    Op::construct_output_parallel_dims(records,
                                       std::get<0>(mapping),
                                       std::get<2>(mapping),
                                       input_idx,
                                       output_idx,
                                       std::get<1>(mapping));
  }
}

/*static*/
void Op::construct_output_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    std::vector<std::pair<int, int>> mappings,
    int input_idx,
    int output_idx) {
  for (std::pair<int, int> const &mapping : mappings) {
    Op::construct_output_parallel_dims(
        records, mapping.first, mapping.second, input_idx, output_idx);
  }
}

/*static*/
void Op::construct_output_parallel_dims(
    std::vector<ParallelDimMappingRecord> &records,
    int input_dim,
    int output_dim,
    int input_idx,
    int output_idx,
    tl::optional<MappingOperation> operation) {
  records.push_back(ParallelDimMappingRecord::input_output_record(
      input_idx, input_dim, output_idx, output_dim, operation));
}

void Op::register_output_parallel_dims(
    std::vector<std::pair<int, int>> mappings, int input_idx, int output_idx) {
  Op::construct_output_parallel_dims(
      *this->parallel_dims_mapping, mappings, input_idx, output_idx);
}

void Op::register_output_parallel_dims(
    std::vector<std::tuple<int, MappingOperation, int>> mappings,
    int input_idx,
    int output_idx) {
  Op::construct_output_parallel_dims(
      *this->parallel_dims_mapping, mappings, input_idx, output_idx);
}

void Op::register_output_parallel_dims(
    int input_dim,
    int output_dim,
    int input_idx,
    int output_idx,
    tl::optional<MappingOperation> operation) {
  Op::construct_output_parallel_dims(*this->parallel_dims_mapping,
                                     input_dim,
                                     output_dim,
                                     input_idx,
                                     output_idx,
                                     operation);
}

int Op::get_output_to_input_dim_mapping(const ParallelTensor output,
                                        int output_dim,
                                        const ParallelTensor input) {
  int output_idx = -1, input_idx = -1;
  for (int i = 0; i < numOutputs; i++)
    if (output == outputs[i])
      output_idx = i;
  for (int i = 0; i < numInputs; i++)
    if (input == inputs[i])
      input_idx = i;
  assert(output_idx != -1);
  assert(input_idx != -1);
  for (size_t i = 0; i < parallel_dims_mapping->size(); i++) {
    if ((*parallel_dims_mapping)[i].output_idx != output_idx)
      continue;
    if ((*parallel_dims_mapping)[i].output_dim != output_dim)
      continue;
    if ((*parallel_dims_mapping)[i].input_idx != input_idx)
      continue;
    // Check validness
    assert((*parallel_dims_mapping)[i].weight_idx = -1);
    assert((*parallel_dims_mapping)[i].weight_dim = -1);
    return (*parallel_dims_mapping)[i].input_dim;
  }
  assert(false);
  return -1;
}

int Op::get_output_to_weight_dim_mapping(const ParallelTensor output,
                                         int output_dim,
                                         const ParallelTensor weight) {
  int output_idx = -1, weight_idx = -1;
  for (int i = 0; i < numOutputs; i++)
    if (output == outputs[i])
      output_idx = i;
  for (int i = 0; i < numInputs; i++)
    if (weight == weights[i])
      weight_idx = i;
  assert(output_idx != -1);
  assert(weight_idx != -1);
  for (size_t i = 0; i < parallel_dims_mapping->size(); i++) {
    if ((*parallel_dims_mapping)[i].output_idx != output_idx)
      continue;
    if ((*parallel_dims_mapping)[i].output_dim != output_dim)
      continue;
    if ((*parallel_dims_mapping)[i].weight_idx != weight_idx)
      continue;
    // Check validness
    assert((*parallel_dims_mapping)[i].input_idx = -1);
    assert((*parallel_dims_mapping)[i].input_dim = -1);
    return (*parallel_dims_mapping)[i].weight_dim;
  }
  assert(false);
  return -1;
}

bool Op::check_output_input_weight_parallel_dims(bool allocate_weights) const {
  // if (!allocate_weights) {
  //   assert(this->numWeights == 0);
  // }

  for (ParallelDimMappingRecord const &record : *parallel_dims_mapping) {
    assert(record.input_idx < this->numInputs);
    assert(record.input_dim < this->inputs[record.input_idx]->num_dims);
    ParallelDim const &input_dim =
        inputs[record.input_idx]->dims[record.input_dim];
    /* assert (input_dim.degree != ParallelDim::UNKNOWN_DEGREE); */
    /* assert (input_dim.parallel_idx != ParallelDim::UNKNOWN_INDEX); */

    ParallelDim other_dim;
    switch (record.get_type()) {
      case MappingRecordType::INPUT_OUTPUT:
        assert(record.output_idx < this->numOutputs);
        assert(record.output_dim < this->outputs[record.output_idx]->num_dims);
        other_dim = outputs[record.output_idx]->dims[record.output_dim];
        break;
      case MappingRecordType::INPUT_WEIGHT:
        if (!allocate_weights) {
          continue;
        }
        if (record.weight_idx >= this->numWeights) {
          // The case where some weights are not used (e.g., no bias for linear)
          continue;
        }
        assert(record.weight_dim < this->weights[record.weight_idx]->num_dims);
        other_dim = weights[record.weight_idx]->dims[record.weight_dim];
        break;
    }

    assert(other_dim.degree == input_dim.degree);
    assert(other_dim.parallel_idx == input_dim.parallel_idx);
  }
  return true;
}

bool Op::check_output_input_weight_same_parallel_is() const {
  assert(numOutputs > 0);
  IndexSpace parallel_is = outputs[0]->parallel_is;
  for (int i = 0; i < numOutputs; i++)
    if (outputs[i]->parallel_is != parallel_is)
      return false;
  for (int i = 0; i < numInputs; i++)
    if (inputs[i]->parallel_is != parallel_is)
      return false;
  for (int i = 0; i < numWeights; i++)
    if (weights[i]->parallel_is != parallel_is)
      return false;
  return true;
}

bool Op::check_output_input_weight_same_machine_view() const {
  assert(numOutputs > 0);
  MachineView machine_view = outputs[0]->machine_view;
  for (int i = 0; i < numOutputs; i++)
    if (outputs[i]->machine_view != machine_view)
      return false;
  for (int i = 0; i < numInputs; i++)
    if (inputs[i]->machine_view != machine_view)
      return false;
  for (int i = 0; i < numWeights; i++)
    if (weights[i]->machine_view != machine_view)
      return false;
  return true;
}

void Op::set_argumentmap_for_init(FFModel const &ff, ArgumentMap &argmap) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, this->parallel_is);
  switch (domain.get_dim()) {
#ifdef FF_USE_NCCL
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    MachineView view = outputs[0]->machine_view;                               \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      FFHandler handle = ff.handlers[view.get_device_id(*it)];                 \
      if (ff.config.computationMode == COMP_MODE_TRAINING &&                   \
          op_type == OP_WEIGHT) {                                              \
        ncclComm_t *nccl_comms = ff.find_nccl_comms(view);                     \
        handle.ncclComm = nccl_comms[idx++];                                   \
      }                                                                        \
      argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));         \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
#else
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    MachineView view = outputs[0]->machine_view;                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      FFHandler handle = ff.handlers[view.get_device_id(*it)];                 \
      argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));         \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
#endif
    default:
      assert(false);
  }
}

void Op::set_opmeta_from_futuremap(FFModel const &ff, FutureMap const &fm) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      meta[idx++] = fm.get_result<OpMeta *>(*it);                              \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void Op::set_argumentmap_for_forward(FFModel const &ff, ArgumentMap &argmap) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      OpMeta *mp = meta[idx++];                                                \
      argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta *)));              \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void Op::set_argumentmap_for_backward(FFModel const &ff, ArgumentMap &argmap) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> rect = domain;                                                   \
    int idx = 0;                                                               \
    for (PointInRectIterator<DIM> it(rect); it(); it++) {                      \
      OpMeta *mp = meta[idx++];                                                \
      argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta *)));              \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

bool Op::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
    case PM_OP_TYPE:
      *value = (int)op_type;
      return true;
    case PM_NUM_INPUTS:
      *value = numInputs;
      return true;
    case PM_NUM_OUTPUTS:
      *value = numOutputs;
      return true;
    default:
      return false;
  }
}

bool Op::get_tensor_parameter(TNParameter tnp,
                              DIMParameter dim,
                              int *value) const {
  if (tnp >= INPUT_0 && tnp <= INPUT_5)
    return get_input_parameter(tnp, dim, value);
  if (tnp >= WEIGHT_0 && tnp <= WEIGHT_5)
    return get_weight_parameter(tnp, dim, value);
  return false;
}

bool Op::get_input_parameter(TNParameter tnp,
                             DIMParameter dim,
                             int *value) const {
  int inputIdx = 0, dimIdx = 0;
  assert(tnp <= INPUT_5 && tnp >= INPUT_0);
  inputIdx = tnp - INPUT_0;
  if (inputIdx >= numInputs)
    return false;
  switch (dim) {
    case DIM_3:
      dimIdx++;
    case DIM_2:
      dimIdx++;
    case DIM_1:
      dimIdx++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = inputs[inputIdx]->num_dims;
      return true;
    default:
      return false;
  }
  if (dimIdx >= inputs[inputIdx]->num_dims)
    return false;
  *value = inputs[inputIdx]->dims[dimIdx].size;
  return true;
}

bool Op::get_weight_parameter(TNParameter tnp,
                              DIMParameter dim,
                              int *value) const {
  int weightIdx = 0, dimIdx = 0;
  assert(tnp <= WEIGHT_5 && tnp >= WEIGHT_0);
  weightIdx = tnp - WEIGHT_0;
  if (weightIdx >= numWeights)
    return false;
  switch (dim) {
    case DIM_3:
      dimIdx++;
    case DIM_2:
      dimIdx++;
    case DIM_1:
      dimIdx++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = weights[weightIdx]->num_dims;
      return true;
    default:
      return false;
  }
  if (dimIdx >= weights[weightIdx]->num_dims)
    return false;
  *value = weights[weightIdx]->dims[dimIdx].size;
  return true;
}

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  throw std::runtime_error(
      "No overload of get_params_hash defined for op type " +
      get_operator_type_name(this->op_type));
}

ParallelConfig Op::get_data_parallel_config(FFModel const &ff) const {
  return get_basic_data_parallel_config(
      ff.config.workersPerNode * ff.config.numNodes, this->get_dimension());
}

ParallelConfig get_basic_data_parallel_config(int num_parts, int dims) {
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = dims;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = i;
  return pc;
}

ParallelConfig Op::get_random_parallel_config(FFModel const &ff) const {
  std::vector<int> candidates;
  int batch_size = outputs[0]->dims[outputs[0]->num_dims - 1].size;
  for (int i = 1; i <= ff.config.workersPerNode; i++)
    if (ff.config.workersPerNode % i == 0) {
      if (batch_size % i != 0)
        continue;
      candidates.push_back(i);
    }
  for (int i = 1; i <= ff.config.numNodes; i++)
    if (ff.config.numNodes % i == 0) {
      if (batch_size % (i * ff.config.workersPerNode) != 0)
        continue;
      candidates.push_back(i * ff.config.workersPerNode);
    }
  assert(candidates.size() > 0);
  int idx = std::rand() % candidates.size();
  int num_parts = candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  int total_num_devices = ff.config.workersPerNode * ff.config.numNodes;
  int start_idx = std::rand() % (total_num_devices - num_parts + 1);
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

int Op::get_dimension() const {
  return this->outputs[0]->num_dims;
}

ParallelConfig ParallelConfig::change_data_parallel_dimensionality(
    int new_dimensionality) const {
  ParallelConfig pc = *this;
  assert(this->is_data_parallel());
  assert(new_dimensionality <= MAX_TENSOR_DIM);
  assert(new_dimensionality > 0);

  for (int i = 0; i < new_dimensionality - 1; i++) {
    pc.dim[i] = 1;
  }
  pc.dim[new_dimensionality - 1] = this->dim[this->nDims - 1];
  pc.nDims = new_dimensionality;

  return pc;
}

bool Op::is_adoptable_parallel_config(FFModel const &ff,
                                      ParallelConfig const &pc) const {
  if (this->is_valid_parallel_config(ff, pc)) {
    return true;
  }

  if (pc.is_data_parallel()) {
    ParallelConfig adopted_pc =
        pc.change_data_parallel_dimensionality(this->outputs[0]->num_dims);
    if (this->is_valid_parallel_config(ff, adopted_pc)) {
      return true;
    }
  }

  return false;
}

bool Op::is_valid_parallel_config(FFModel const &ff,
                                  ParallelConfig const &pc) const {
  // By default only data parallelism is allowed
  // Check dim match
  if (pc.nDims != this->get_dimension())
    return false;
  for (int i = 0; i < pc.nDims - 1; i++)
    if (pc.dim[i] != 1)
      return false;
  return true;
}

Domain Op::get_output_tensor_shape(ParallelConfig const &pc,
                                   int output_idx,
                                   int part_idx) const {
  assert(output_idx < numOutputs);
  Domain d;
  d.dim = outputs[output_idx]->num_dims;
  // Assume pc dim matches output dim
  assert(d.dim == pc.nDims);
  for (int i = 0; i < d.dim; i++) {
    // Assume an equal partitioning
    assert(outputs[output_idx]->dims[i].size % pc.dim[i] == 0);
    int dim_size = outputs[output_idx]->dims[i].size / pc.dim[i];
    d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
    d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    part_idx = part_idx / pc.dim[i];
  }
  assert(part_idx == 0);
  return d;
}

Domain Op::get_input_tensor_shape(ParallelConfig const &pc,
                                  int input_idx,
                                  int part_idx) const {
  assert(input_idx < numInputs);
  Domain d;
  d.dim = inputs[input_idx]->num_dims;
  if (pc.nDims == d.dim) {
    for (int i = 0; i < d.dim; i++) {
      // Assume an equal partitioning
      assert(inputs[input_idx]->dims[i].size % pc.dim[i] == 0);
      int dim_size = inputs[input_idx]->dims[i].size / pc.dim[i];
      d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
      part_idx = part_idx / pc.dim[i];
    }
  } else {
    // Require data parallel when dims mismatch
    for (int i = 0; i < pc.nDims; i++)
      if (i != pc.nDims - 2) {
        assert(pc.dim[i] == 1);
      }
    for (int i = 0; i < d.dim - 1; i++) {
      int dim_size = inputs[input_idx]->dims[i].size;
      d.rect_data[i] = 0;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    }
    // Assume an equal partitioning
    assert(inputs[input_idx]->dims[d.dim - 2].size % pc.dim[pc.nDims - 2] == 0);
    assert(part_idx < pc.dim[pc.nDims - 2]);
    int dim_size =
        inputs[input_idx]->dims[d.dim - 2].size / pc.dim[pc.nDims - 2];
    d.rect_data[d.dim - 1] = part_idx * dim_size;
    d.rect_data[2 * d.dim - 1] = d.rect_data[d.dim - 1] + dim_size - 1;
    part_idx = part_idx / pc.dim[pc.nDims - 1];
  }
  assert(part_idx == 0);
  return d;
}

Domain Op::get_weight_tensor_shape(ParallelConfig const &pc,
                                   int weight_idx,
                                   int part_idx) const {
  // Default data parallel weight replication
  assert(weight_idx < numWeights);
  Domain d;
  d.dim = weights[weight_idx]->num_dims;
  for (int i = 0; i < d.dim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i + d.dim] = weights[weight_idx]->dims[i].size /
                                 weights[weight_idx]->dims[i].degree -
                             1;
  }
  return d;
}

void Op::solve_parallel_dim_mappings(
    std::vector<ParallelDim const *> const &inputs,
    std::vector<ParallelDim *> const &weights,
    std::vector<ParallelDim *> const &outputs) const {
  FlexFlow::solve_parallel_dim_mappings(
      *this->parallel_dims_mapping, inputs, weights, outputs);
}

void solve_parallel_dim_mappings(
    std::vector<ParallelDimMappingRecord> const &mapping,
    std::vector<ParallelDim const *> const &inputs,
    std::vector<ParallelDim *> const &weights,
    std::vector<ParallelDim *> const &outputs) {
  for (ParallelDimMappingRecord const &record : mapping) {
    ParallelDim const &input_dim = inputs[record.input_idx][record.input_dim];

    switch (record.get_type()) {
      case MappingRecordType::INPUT_OUTPUT: {
        if (record.output_idx >= outputs.size() ||
            outputs[record.output_idx] == nullptr) {
          continue;
        }

        ParallelDim &output_dim = outputs[record.output_idx][record.output_dim];
        output_dim.degree = input_dim.degree;
        output_dim.parallel_idx = input_dim.parallel_idx;

        if (output_dim.is_replica_dim) {
          output_dim.size = input_dim.degree;
        }
      } break;
      case MappingRecordType::INPUT_WEIGHT: {
        if (record.weight_idx >= weights.size() ||
            weights[record.weight_idx] == nullptr) {
          continue;
        }

        ParallelDim &weight_dim = weights[record.weight_idx][record.weight_dim];
        weight_dim.degree = input_dim.degree;
        weight_dim.parallel_idx = input_dim.parallel_idx;

        if (weight_dim.is_replica_dim) {
          weight_dim.size = input_dim.degree;
        }
      } break;
    }
  }
}

std::unordered_map<int, int> output_to_input_mapping(
    std::vector<ParallelDimMappingRecord> const &mapping) {
  std::unordered_map<int, int> dim_mapping;
  for (ParallelDimMappingRecord const &record : mapping) {
    if (record.get_type() == MappingRecordType::INPUT_OUTPUT) {
      dim_mapping[record.output_dim] = record.input_dim;
    }
  }

  return dim_mapping;
}

std::unordered_map<int, int> input_to_output_mapping(
    std::vector<ParallelDimMappingRecord> const &mapping) {
  std::unordered_map<int, int> dim_mapping;
  for (ParallelDimMappingRecord const &record : mapping) {
    if (record.get_type() == MappingRecordType::INPUT_OUTPUT) {
      dim_mapping[record.input_dim] = record.output_dim;
    }
  }

  return dim_mapping;
}

#ifdef FF_USE_NCCL
ncclUniqueId
    Op::get_nccl_unique_id_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  ncclUniqueId ncclId;
  checkNCCL(ncclGetUniqueId(&ncclId));
  return ncclId;
}

ncclComm_t Op::init_nccl_comms_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  // Must be an index space launch
  assert(task->is_index_space);
  ncclUniqueId ncclId = *((ncclUniqueId const *)task->args);
  int allRanks = task->index_domain.get_volume();
  assert(task->index_domain.contains(task->index_point));
  int myRank = 0;
  for (Domain::DomainPointIterator it(task->index_domain); it; it++, myRank++) {
    if (it.p == task->index_point)
      break;
  }
  ncclComm_t ncclComm;
  checkNCCL(ncclCommInitRank(&ncclComm, allRanks, ncclId, myRank));
  // fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //     ncclComm, allRanks, myRank, ncclId);
  return ncclComm;
}
#endif

Op::Op(FFModel &model,
       OperatorType op_type,
       char const *name,
       int numInputs,
       int numWeights,
       bool allocate_weights,
       int numOutputs,
       const ParallelTensor input1,
       const ParallelTensor input2,
       const ParallelTensor input3,
       const ParallelTensor input4)
    : Op(model,
         op_type,
         name,
         numInputs,
         allocate_weights ? numWeights : 0,
         numOutputs,
         input1,
         input2,
         input3,
         input4) {}

Op::Op(FFModel &model,
       OperatorType _op_type,
       char const *_name,
       int _numInputs,
       int _numWeights,
       int _numOutputs,
       const ParallelTensor _input1,
       const ParallelTensor _input2,
       const ParallelTensor _input3,
       const ParallelTensor _input4)
    : op_type(_op_type), op_guid(model.op_global_guid++), numInputs(_numInputs),
      numWeights(_numWeights), numOutputs(_numOutputs),
      profiling(model.config.profiling) {
  for (int i = 0; i < MAX_NUM_INPUTS; i++)
    inputs[i] = NULL;
  std::vector<ParallelTensor> tensors;
  tensors.push_back(_input1);
  tensors.push_back(_input2);
  tensors.push_back(_input3);
  tensors.push_back(_input4);
  std::string pcname;
  if (_name == NULL) {
    pcname = get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(op_guid);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < numInputs; i++) {
    assert(tensors[i] != NULL);
    inputs[i] = tensors[i];
  }
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    // resetInputGrads[i] = true;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = NULL;
  }
  for (int i = 0; i < MAX_NUM_WORKERS; i++)
    meta[i] = NULL;
  parallel_dims_mapping = new std::vector<ParallelDimMappingRecord>();
}

Op::Op(FFModel &model,
       OperatorType _op_type,
       char const *_name,
       int _numInputs,
       int _numWeights,
       int _numOutputs,
       ParallelTensor const *_inputs)
    : op_type(_op_type), op_guid(model.op_global_guid++), numInputs(_numInputs),
      numWeights(_numWeights), numOutputs(_numOutputs),
      profiling(model.config.profiling) {
  std::string pcname;
  if (_name == NULL) {
    pcname = get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(op_guid);
  assert(pcname.length() < MAX_OPNAME);
  assert(numInputs <= MAX_NUM_INPUTS);
  assert(numWeights <= MAX_NUM_WEIGHTS);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < numInputs + numWeights; i++) {
    if (i < numInputs) {
      // Activation
      inputs[i] = _inputs[i];
    } else {
      // Weight
      weights[i - numInputs] = _inputs[i];
    }
  }
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    // resetInputGrads[i] = true;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = NULL;
  }
  for (int i = 0; i < MAX_NUM_WORKERS; i++)
    meta[i] = NULL;
  parallel_dims_mapping = new std::vector<ParallelDimMappingRecord>();
}

Op::Op(FFModel &model,
       OperatorType type,
       char const *name,
       int numWeights,
       int numOutputs,
       std::vector<ParallelTensor> const &inputs)
    : Op(model,
         type,
         name,
         inputs.size(),
         numWeights,
         numOutputs,
         inputs.data()) {}

bool Op::is_parallel_op() const {
  return false;
}

bool Op::can_inplace_output() {
  return false;
}

bool Op::has_inplace_output() {
  return false;
}

void Op::do_inplace_output() {
  assert(false);
}

void Op::map_output_tensors(FFModel &ff) {
  for (int i = 0; i < numOutputs; i++)
    ff.map_tensor(outputs[i], this);
}

tl::optional<RecordFormatter> Op::as_dot() const {
  if (this->numOutputs != 1) {
    return tl::nullopt;
  }

  ParallelTensor const &output = this->outputs[0];
  return output->get_shape().as_dot();
}

ParallelTensor Op::get_parameter(int index) {
  assert(index < numWeights);
  return weights[index];
}

void Op::serialize(Legion::Serializer &serializer) const {
  fprintf(stderr,
          "The following operator type is currently not supported"
          " for graph serialization: %s\n"
          "Report the issue to the FlexFlow developers",
          optype_to_string(this->op_type).c_str());
  assert(false && "This op does not support serialization");
}

Op *Op::materialize(FFModel &ff,
                    ParallelTensor inputs[],
                    int num_inputs) const {
  fprintf(stderr,
          "The following operator type is currently not supported"
          " for layer materialization: %s\n"
          "Report the issue to the FlexFlow developers",
          optype_to_string(this->op_type).c_str());
  assert(false && "This op does not support materialization");
}

void Op::zero_grad(FFModel const &ff) {
  // Do nothing for input and weight
  if (op_type == OP_INPUT || op_type == OP_WEIGHT)
    return;
  Runtime *runtime = ff.config.lg_hlr;
  Context ctx = ff.config.lg_ctx;
  ArgumentMap argmap;
  ZeroInitMeta meta;
  meta.op_ptr = this;
  meta.num_regions = numWeights + numOutputs;
  assert(meta.num_regions <= ZeroInitMeta::MAX_NUM_REGIONS);
  IndexSpace parallel_is = IndexSpace::NO_SPACE;
  for (int i = 0; i < numWeights; i++) {
    meta.data_types[i] = weights[i]->data_type;
    if (parallel_is == IndexSpace::NO_SPACE)
      parallel_is = weights[i]->parallel_is;
    else
      assert(parallel_is == weights[i]->parallel_is);
  }
  for (int i = 0; i < numOutputs; i++) {
    meta.data_types[i + numWeights] = outputs[i]->data_type;
    if (parallel_is == IndexSpace::NO_SPACE)
      parallel_is = outputs[i]->parallel_is;
    else
      assert(parallel_is == outputs[i]->parallel_is);
  }
  IndexLauncher launcher(ZERO_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(&meta, sizeof(ZeroInitMeta)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(RegionRequirement(weights[i]->part_grad,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      weights[i]->region_grad));
    launcher.add_field(i, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part_grad,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region_grad));
    // LogicalRegion lr = outputs[i]->region_grad;
    // printf("zero_grad:output[%d]: region(%d,%d,%d)\n", i,
    // lr.get_index_space().get_id(), lr.get_field_space().get_id(),
    // lr.get_tree_id());
    launcher.add_field(i + numWeights, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
