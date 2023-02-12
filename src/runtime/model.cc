/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/model.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif
#include "flexflow/ffconst_utils.h"
#include "flexflow/graph.h"
#include "flexflow/mapper.h"
#include "flexflow/ops/aggregate.h"
#include "flexflow/ops/aggregate_spec.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/batch_matmul.h"
#include "flexflow/ops/batch_norm.h"
#include "flexflow/ops/cache.h"
#include "flexflow/ops/cast.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/fused.h"
#include "flexflow/ops/gather.h"
#include "flexflow/ops/groupby.h"
#include "flexflow/ops/layer_norm.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/noop.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/reduce.h"
#include "flexflow/ops/reshape.h"
#include "flexflow/ops/reverse.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/split.h"
#include "flexflow/ops/topk.h"
#include "flexflow/ops/transpose.h"
#include "flexflow/parallel_ops/combine.h"
#include "flexflow/parallel_ops/fused_parallel_op.h"
#include "flexflow/parallel_ops/partition.h"
#include "flexflow/parallel_ops/reduction.h"
#include "flexflow/parallel_ops/replicate.h"
#include "flexflow/substitution.h"
#include "flexflow/utils/random_utils.h"
#include "flexflow/utils/test_utils.h"
#include "legion/legion_utilities.h"
#include <dirent.h>
#include <queue>
#include <unordered_set>

namespace FlexFlow {

using namespace Legion;

LegionRuntime::Logger::Category log_model("Model");
LegionRuntime::Logger::Category log_measure("measure");

Op::Op(FFModel &model,
       OperatorType otype,
       DataType dtype,
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
         otype,
         dtype,
         name,
         numInputs,
         allocate_weights ? numWeights : 0,
         numOutputs,
         input1,
         input2,
         input3,
         input4) {}

Op::Op(FFModel &model,
       OperatorType _otype,
       DataType _dtype,
       char const *_name,
       int _numInputs,
       int _numWeights,
       int _numOutputs,
       const ParallelTensor _input1,
       const ParallelTensor _input2,
       const ParallelTensor _input3,
       const ParallelTensor _input4)
    : op_type(_otype), data_type(_dtype), op_guid(model.op_global_guid++),
      numInputs(_numInputs), numWeights(_numWeights), numOutputs(_numOutputs),
      profiling(model.config.profiling) {
  for (int i = 0; i < MAX_NUM_INPUTS; i++) {
    inputs[i] = NULL;
  }
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
    outputs[i] = nullptr;
  }
  for (int i = 0; i < MAX_NUM_WORKERS; i++) {
    meta[i] = nullptr;
  }
  parallel_dims_mapping = new std::vector<ParallelDimMappingRecord>();
}

Op::Op(FFModel &model,
       OperatorType _otype,
       DataType _dtype,
       char const *_name,
       int _numInputs,
       int _numWeights,
       int _numOutputs,
       ParallelTensor const *_inputs)
    : op_type(_otype), data_type(_dtype), op_guid(model.op_global_guid++),
      numInputs(_numInputs), numWeights(_numWeights), numOutputs(_numOutputs),
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
  for (int i = 0; i < MAX_NUM_WORKERS; i++) {
    meta[i] = NULL;
  }
  parallel_dims_mapping = new std::vector<ParallelDimMappingRecord>();
}

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
  for (int i = 0; i < numOutputs; i++) {
    ff.map_tensor(outputs[i], this);
  }
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
          "Report the issue to the FlexFlow developers\n",
          get_operator_type_name(this->op_type).c_str());
  assert(false && "This op does not support serialization");
}

Op *Op::materialize(FFModel &ff,
                    ParallelTensor inputs[],
                    int num_inputs) const {
  fprintf(stderr,
          "The following operator type is currently not supported"
          " for layer materialization: %s\n"
          "Report the issue to the FlexFlow developers\n",
          get_operator_type_name(this->op_type).c_str());
  assert(false && "This op does not support materialization");
}

void Op::zero_grad(FFModel const &ff) {
  // Do nothing for input and weight
  if (op_type == OP_INPUT || op_type == OP_WEIGHT) {
    return;
  }
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
    if (parallel_is == IndexSpace::NO_SPACE) {
      parallel_is = weights[i]->parallel_is;
    } else {
      assert(parallel_is == weights[i]->parallel_is);
    }
  }
  for (int i = 0; i < numOutputs; i++) {
    meta.data_types[i + numWeights] = outputs[i]->data_type;
    if (parallel_is == IndexSpace::NO_SPACE) {
      parallel_is = outputs[i]->parallel_is;
    } else {
      assert(parallel_is == outputs[i]->parallel_is);
    }
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

ParallelConfig Op::get_data_parallel_config(FFModel const &ff) const {
  return get_basic_data_parallel_config(
      ff.config.workersPerNode * ff.config.numNodes, this->get_dimension());
}

ParallelConfig get_basic_data_parallel_config(int num_parts, int dims) {
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = dims;
  for (int i = 0; i < pc.nDims; i++) {
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  }
  for (int i = 0; i < num_parts; i++) {
    pc.device_ids[i] = i;
  }
  return pc;
}

ParallelConfig Op::get_random_parallel_config(FFModel const &ff) const {
  std::vector<int> candidates;
  int batch_size = outputs[0]->dims[outputs[0]->num_dims - 1].size;
  for (int i = 1; i <= ff.config.workersPerNode; i++) {
    if (ff.config.workersPerNode % i == 0) {
      if (batch_size % i != 0) {
        continue;
      }
      candidates.push_back(i);
    }
  }
  for (int i = 1; i <= ff.config.numNodes; i++) {
    if (ff.config.numNodes % i == 0) {
      if (batch_size % (i * ff.config.workersPerNode) != 0) {
        continue;
      }
      candidates.push_back(i * ff.config.workersPerNode);
    }
  }
  assert(candidates.size() > 0);
  int idx = std::rand() % candidates.size();
  int num_parts = candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  for (int i = 0; i < pc.nDims; i++) {
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  }
  int total_num_devices = ff.config.workersPerNode * ff.config.numNodes;
  int start_idx = std::rand() % (total_num_devices - num_parts + 1);
  for (int i = 0; i < num_parts; i++) {
    pc.device_ids[i] = start_idx + i;
  }
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
  if (pc.nDims != this->get_dimension()) {
    return false;
  }
  for (int i = 0; i < pc.nDims - 1; i++) {
    if (pc.dim[i] != 1) {
      return false;
    }
  }
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
    for (int i = 0; i < pc.nDims; i++) {
      if (i != pc.nDims - 2) {
        assert(pc.dim[i] == 1);
      }
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
    if (it.p == task->index_point) {
      break;
    }
  }
  ncclComm_t ncclComm;
  checkNCCL(ncclCommInitRank(&ncclComm, allRanks, ncclId, myRank));
  // fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //     ncclComm, allRanks, myRank, ncclId);
  return ncclComm;
}
#endif

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
  for (int i = 0; i < numOutputs; i++) {
    if (output == outputs[i]) {
      output_idx = i;
    }
  }
  for (int i = 0; i < numInputs; i++) {
    if (input == inputs[i]) {
      input_idx = i;
    }
  }
  assert(output_idx != -1);
  assert(input_idx != -1);
  for (size_t i = 0; i < parallel_dims_mapping->size(); i++) {
    if ((*parallel_dims_mapping)[i].output_idx != output_idx) {
      continue;
    }
    if ((*parallel_dims_mapping)[i].output_dim != output_dim) {
      continue;
    }
    if ((*parallel_dims_mapping)[i].input_idx != input_idx) {
      continue;
    }
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
  for (int i = 0; i < numOutputs; i++) {
    if (output == outputs[i]) {
      output_idx = i;
    }
  }
  for (int i = 0; i < numInputs; i++) {
    if (weight == weights[i]) {
      weight_idx = i;
    }
  }
  assert(output_idx != -1);
  assert(weight_idx != -1);
  for (size_t i = 0; i < parallel_dims_mapping->size(); i++) {
    if ((*parallel_dims_mapping)[i].output_idx != output_idx) {
      continue;
    }
    if ((*parallel_dims_mapping)[i].output_dim != output_dim) {
      continue;
    }
    if ((*parallel_dims_mapping)[i].weight_idx != weight_idx) {
      continue;
    }
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
  for (int i = 0; i < numOutputs; i++) {
    if (outputs[i]->parallel_is != parallel_is) {
      return false;
    }
  }
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i]->parallel_is != parallel_is) {
      return false;
    }
  }
  for (int i = 0; i < numWeights; i++) {
    if (weights[i]->parallel_is != parallel_is) {
      return false;
    }
  }
  return true;
}

bool Op::check_output_input_weight_same_machine_view() const {
  assert(numOutputs > 0);
  MachineView machine_view = outputs[0]->machine_view;
  for (int i = 0; i < numOutputs; i++) {
    if (outputs[i]->machine_view != machine_view) {
      return false;
    }
  }
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i]->machine_view != machine_view) {
      return false;
    }
  }
  for (int i = 0; i < numWeights; i++) {
    if (weights[i]->machine_view != machine_view) {
      return false;
    }
  }
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
  if (tnp >= INPUT_0 && tnp <= INPUT_5) {
    return get_input_parameter(tnp, dim, value);
  }
  if (tnp >= WEIGHT_0 && tnp <= WEIGHT_5) {
    return get_weight_parameter(tnp, dim, value);
  }
  return false;
}

bool Op::get_input_parameter(TNParameter tnp,
                             DIMParameter dim,
                             int *value) const {
  int inputIdx = 0, dimIdx = 0;
  assert(tnp <= INPUT_5 && tnp >= INPUT_0);
  inputIdx = tnp - INPUT_0;
  if (inputIdx >= numInputs) {
    return false;
  }
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
  if (dimIdx >= inputs[inputIdx]->num_dims) {
    return false;
  }
  *value = inputs[inputIdx]->dims[dimIdx].size;
  return true;
}

bool Op::get_weight_parameter(TNParameter tnp,
                              DIMParameter dim,
                              int *value) const {
  int weightIdx = 0, dimIdx = 0;
  assert(tnp <= WEIGHT_5 && tnp >= WEIGHT_0);
  weightIdx = tnp - WEIGHT_0;
  if (weightIdx >= numWeights) {
    return false;
  }
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
  if (dimIdx >= weights[weightIdx]->num_dims) {
    return false;
  }
  *value = weights[weightIdx]->dims[dimIdx].size;
  return true;
}

OpMeta::OpMeta(FFHandler _handle) : handle(_handle), profiling(false) {
  for (int i = 0; i < MAX_NUM_INPUTS; i++) {
    trainableInputs[i] = true;
  }
  for (int i = 0; i < MAX_NUM_INPUTS; i++) {
    input_type[i] = DT_NONE;
  }
  for (int i = 0; i < MAX_NUM_WEIGHTS; i++) {
    weight_type[i] = DT_NONE;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    output_type[i] = DT_NONE;
  }
}

OpMeta::OpMeta(FFHandler _handle, Op const *op) : OpMeta(_handle) {
  for (int i = 0; i < op->numInputs; i++) {
    input_type[i] = op->inputs[i]->data_type;
  }
  for (int i = 0; i < op->numWeights; i++) {
    weight_type[i] = op->weights[i]->data_type;
  }
  for (int i = 0; i < op->numOutputs; i++) {
    output_type[i] = op->outputs[i]->data_type;
  }
}

FFModel::FFModel(FFConfig &_config)
    : op_global_guid(OP_GUID_FIRST_VALID),
      layer_global_guid(LAYER_GUID_FIRST_VALID),
      tensor_global_guid(TENSOR_GUID_FIRST_VALID),
      parallel_tensor_global_guid(PARALLEL_TENSOR_GUID_FIRST_VALID),
      node_global_guid(NODE_GUID_FIRST_VALID), config(_config), optimizer(NULL),
      loss_op(NULL), metrics_op(NULL), simulator(NULL) {
  this->search = new PCG::SearchHelper(this);
  this->graph_search = new PCG::GraphSearchHelper(this);

  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
  // Register machine views
  register_all_machine_views(config.numNodes,
                             config.workersPerNode,
                             config.cpusPerNode,
                             all_valid_views);
  metrics_input = -1;
  // Load strategy file
  // Create field space
  {
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  // Build training dataset
  // if (config.datasetPath.length() == 0) {
  //  dataLoader = NULL;
  //} else {
  //  dataLoader = new DataLoader(config.datasetPath);
  //}

  ArgumentMap argmap;
  Rect<1> task_rect(Point<1>(0),
                    Point<1>(config.workersPerNode * config.numNodes - 1));
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);

  // int rank = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    FFInitInfo info;
    // info.myRank = rank++;
    // info.allRanks = config.workersPerNode * config.numNodes;
    info.workSpaceSize = config.workSpaceSize;
    info.allowTensorOpMathConversion = config.allow_tensor_op_math_conversion;
    argmap.set_point(*it, TaskArgument(&info, sizeof(FFInitInfo)));
  }

  // Init CUDA library on each worker
  IndexLauncher initLauncher(FF_INIT_TASK_ID,
                             task_is,
                             TaskArgument(NULL, 0),
                             argmap,
                             Predicate::TRUE_PRED,
                             false /*must*/,
                             0 /*mapper_id*/,
                             FFConfig::DataParallelism_GPU);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    handlers[idx++] = fm.get_result<FFHandler>(*it);
  }
}

void FFModel::clear_graph_search_cache() {
  this->graph_search->clear_cache();
  this->search->clear_cache();
}

#ifdef FF_USE_NCCL
ncclComm_t *FFModel::find_nccl_comms(MachineView const &view) const {
  auto const &it = view_hash_to_nccl_comms.find(view.hash());
  if (it == view_hash_to_nccl_comms.end()) {
    assert(config.computationMode == COMP_MODE_INFERENCE);
    return NULL;
  } else {
    return it->second;
  }
}
#endif

template <int NDIM>
Tensor FFModel::create_constant(int const dims[],
                                float value,
                                DataType data_type) {
  // FIXME: currently create gradients for constants since the current auto grad
  // algorithm computes gradients for all operators
  Tensor tensor = create_tensor<NDIM>(
      dims, data_type, NULL /*owner_op*/, false /*create_grad*/);
  tensor->initializer = new ConstantInitializer(value);
  return tensor;
#ifdef DEADCODE
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  assert(false);
  ArgumentMap argmap;
  IndexLauncher launcher(CONSTANT_INIT_TASK_ID,
                         tensor->parallel_is,
                         TaskArgument(init, sizeof(ConstantInitializer)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false,
                         0,
                         tensor->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(tensor->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    tensor->region));
  launcher.add_field(0, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return tensor;
#endif
}

PCG::Node FFModel::new_node(Op *op) {
  PCG::Node ret;
  ret.guid = this->node_global_guid++;
  ret.ptr = op;

  return ret;
}

Tensor FFModel::create_tensor(int numdim,
                              int const dims[],
                              DataType data_type,
                              Layer const *layer,
                              int idx,
                              bool create_grad) {
  switch (numdim) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return create_tensor<DIM>(dims, data_type, layer, idx, create_grad);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported dim!");
  }
}

ParallelTensor FFModel::create_parallel_tensor(int numdim,
                                               const ParallelDim dims[],
                                               DataType data_type,
                                               Op const *op,
                                               int idx,
                                               bool create_grad,
                                               size_t input_tensor_guid) {
  switch (numdim) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return create_parallel_tensor<DIM>(                                        \
        dims, data_type, op, idx, create_grad, input_tensor_guid);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported dim!");
  }
}

Tensor FFModel::create_tensor_legion_ordering(int numdim,
                                              int const dims[],
                                              DataType data_type,
                                              Layer const *layer,
                                              int idx,
                                              bool create_grad) {
  int c_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    c_dims[i] = dims[numdim - 1 - i];
  }
  return create_tensor(numdim, c_dims, data_type, layer, idx, create_grad);
}

ParallelTensor
    FFModel::create_parallel_tensor_legion_ordering(int numdim,
                                                    const ParallelDim dims[],
                                                    DataType data_type,
                                                    Op const *op,
                                                    int idx,
                                                    bool create_grad,
                                                    size_t input_tensor_guid) {
  ParallelDim c_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    c_dims[i] = dims[numdim - 1 - i];
  }
  return create_parallel_tensor(
      numdim, c_dims, data_type, op, idx, create_grad, input_tensor_guid);
}

template <int NDIM>
Tensor FFModel::create_tensor(int const dims[],
                              DataType data_type,
                              Layer const *owner_layer,
                              int owner_idx,
                              bool create_grad) {
  Tensor tensor = new TensorBase();
  tensor->tensor_guid = tensor_global_guid++;
  tensor->data_type = data_type;
  if (owner_layer == NULL) {
    Layer *input_layer = new Layer(this,
                                   OP_INPUT,
                                   data_type,
                                   "input",
                                   0 /*inputs*/,
                                   0 /*weight*/,
                                   1 /*outputs*/,
                                   NULL,
                                   NULL);
    input_layer->outputs[0] = tensor;
    layers.push_back(input_layer);
    tensor->owner_layer = input_layer;
    tensor->owner_idx = 0;
  } else {
    tensor->owner_layer = owner_layer;
    tensor->owner_idx = owner_idx;
  }
  tensor->create_gradients = create_grad;
  tensor->num_dims = NDIM;
  for (int i = 0; i < NDIM; i++) {
    tensor->dims[i] = dims[NDIM - 1 - i];
  }
  return tensor;
}

template <int NDIM>
ParallelTensor FFModel::create_parallel_tensor(const ParallelDim dims[],
                                               DataType data_type,
                                               Op const *owner_op,
                                               int owner_idx,
                                               bool create_grad,
                                               size_t input_tensor_guid) {
  ParallelTensor tensor = new ParallelTensorBase();
  tensor->parallel_tensor_guid = parallel_tensor_global_guid++;
  tensor->data_type = data_type;
  if (owner_op == nullptr) {
    NoOp *input_op = new NoOp(*this, OP_INPUT, input_tensor_guid, tensor);
    operators.push_back(input_op);
    tensor->owner_op = input_op;
    tensor->owner_idx = 0;
  } else {
    tensor->owner_op = owner_op;
    tensor->owner_idx = owner_idx;
  }
  tensor->create_gradients = create_grad;
  tensor->num_dims = NDIM;
  for (int i = 0; i < NDIM; i++) {
    tensor->dims[i] = dims[NDIM - 1 - i];
  }
  assert(tensor->check_valid());
  return tensor;
}

Parameter FFModel::create_weight_legion_ordering(int numdim,
                                                 int const dims[],
                                                 DataType data_type,
                                                 Layer const *layer,
                                                 bool create_grad,
                                                 Initializer *initializer,
                                                 ParameterSyncType sync_type) {
  int c_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    c_dims[i] = dims[numdim - 1 - i];
  }
  return create_weight(
      numdim, c_dims, data_type, layer, create_grad, initializer, sync_type);
}

Parameter FFModel::create_weight(int numdim,
                                 int const dims[],
                                 DataType data_type,
                                 Layer const *owner_layer,
                                 bool create_grad,
                                 Initializer *initializer,
                                 ParameterSyncType sync_type) {
  Parameter p = new TensorBase();
  p->data_type = data_type;
  assert(owner_layer != NULL);
  if (owner_layer == NULL) {
    Layer *weight_layer = new Layer(this,
                                    OP_WEIGHT,
                                    data_type,
                                    NULL,
                                    0 /*inputs*/,
                                    0 /*weights*/,
                                    1 /*outputs*/,
                                    NULL /*in1*/,
                                    NULL /*in2*/);
    layers.push_back(weight_layer);
    p->owner_layer = weight_layer;
    p->owner_idx = 0;
  } else {
    p->owner_layer = owner_layer;
    p->owner_idx = 0;
  }
  p->create_gradients = create_grad;
  p->initializer = initializer;
  p->sync_type = sync_type;
  p->num_dims = numdim;
  for (int i = 0; i < numdim; i++) {
    p->dims[i] = dims[numdim - 1 - i];
  }
  assert(p->get_volume() > 0);
  return p;
}

template <int NDIM>
ParallelParameter FFModel::create_parallel_weight(const ParallelDim dims[],
                                                  DataType data_type,
                                                  Op const *owner_op,
                                                  bool create_grad,
                                                  Initializer *initializer,
                                                  ParameterSyncType sync_type) {
  ParallelParameter p = new ParallelTensorBase();
  p->parallel_tensor_guid = parallel_tensor_global_guid++;
  p->data_type = data_type;
  if (owner_op == NULL) {
    NoOp *weight_op = new NoOp(*this, OP_WEIGHT, p);
    operators.push_back(weight_op);
    p->owner_op = weight_op;
    p->owner_idx = 0;
  } else {
    p->owner_op = owner_op;
  }
  p->create_gradients = create_grad;
  p->initializer = initializer;
  p->sync_type = sync_type;
  p->num_dims = NDIM;
  for (int i = 0; i < NDIM; i++) {
    p->dims[i] = dims[NDIM - 1 - i];
  }
  assert(p->get_volume() > 0);
  assert(p->check_valid());
  return p;
}

ParallelParameter FFModel::create_parallel_weight(int numdim,
                                                  const ParallelDim dims[],
                                                  DataType data_type,
                                                  Op const *owner_op,
                                                  bool create_grad,
                                                  Initializer *initializer,
                                                  ParameterSyncType sync_type) {
  switch (numdim) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return create_parallel_weight<DIM>(                                        \
        dims, data_type, owner_op, create_grad, initializer, sync_type);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported dim!");
  }
}

ParallelParameter FFModel::create_parallel_weight_legion_ordering(
    int numdim,
    const ParallelDim dims[],
    DataType data_type,
    Op const *owner_op,
    bool create_grad,
    Initializer *initializer,
    ParameterSyncType sync_type) {
  ParallelDim c_dims[MAX_TENSOR_DIM];
  std::reverse_copy(dims, dims + numdim, c_dims);

  return this->create_parallel_weight(
      numdim, c_dims, data_type, owner_op, create_grad, initializer, sync_type);
}

void FFModel::map_tensor(ParallelTensor tensor, Op const *op) {
  switch (tensor->num_dims) {
#define DIMFUNC(NDIM)                                                          \
  case NDIM: {                                                                 \
    map_tensor_with_dim<NDIM>(tensor, op);                                     \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dim
      assert(false);
    }
  }
}

// Map tensor using parallelization strategies described in parallel_op
template <int NDIM>
void FFModel::map_tensor_with_dim(ParallelTensor tensor,
                                  Op const *parallel_op) {
  tensor->parallel_is = get_or_create_task_is(tensor);
  assert(tensor->owner_op != NULL);
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Domain task_domain =
      runtime->get_index_space_domain(ctx, tensor->parallel_is);
  switch (task_domain.get_dim()) {
#define DIMFUNC(TDIM)                                                          \
  case TDIM: {                                                                 \
    map_tensor_with_dim2<NDIM, TDIM>(tensor, parallel_op);                     \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      assert(false && "Unsupported Task Dim");
    }
  }
}

template <int NDIM, int TDIM>
void FFModel::map_tensor_with_dim2(ParallelTensor tensor,
                                   Op const *parallel_op) {
  // Step 0: check we are the owner or the owner is NULL
  // in which case set the owner to us
  if (tensor->owner_op == NULL) {
    tensor->owner_op = parallel_op;
    tensor->owner_idx = -1; // meaning tensor is not an output of op
  } else {
    // assert tensor->owner_op == parallel_op or parallel_op == nullptr,
    // which indicates the tensor is not parallelized
    assert(tensor->owner_op == parallel_op || parallel_op == nullptr);
  }
  // Step 1: create regions
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;

  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (tensor->data_type) {
    case DT_HALF:
      allocator.allocate_field(sizeof(half), FID_DATA);
      break;
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int32_t), FID_DATA);
      break;
    case DT_INT64:
      allocator.allocate_field(sizeof(int64_t), FID_DATA);
      break;
    default:
      assert(false);
  }

  Point<NDIM> hi;
  for (int i = 0; i < NDIM; i++) {
    hi[i] = tensor->dims[i].size - 1;
  }
  Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  tensor->region = runtime->create_logical_region(ctx, is, fs);
  if (tensor->create_gradients &&
      config.computationMode == COMP_MODE_TRAINING) {
    tensor->region_grad = runtime->create_logical_region(ctx, is, fs);
  }

  // Step 2: create partitions if parallel_op != NULL
  if (parallel_op != NULL) {
    IndexSpaceT<TDIM> part_is =
        (IndexSpaceT<TDIM>)get_or_create_task_is(tensor);
    // Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
    Transform<NDIM, TDIM> transform;
    Point<NDIM> ext_hi;
    for (int i = 0; i < NDIM; i++) {
      int nparts = tensor->dims[i].degree;
      ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
    }
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        if (tensor->dims[i].parallel_idx == j) {
          transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
        } else {
          transform[i][j] = 0;
        }
      }
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
    tensor->part = runtime->get_logical_partition(ctx, tensor->region, ip);
    if (tensor->create_gradients &&
        config.computationMode == COMP_MODE_TRAINING) {
      tensor->part_grad =
          runtime->get_logical_partition(ctx, tensor->region_grad, ip);
    }
  }
  // Step 3: initialize the tensor
  if (tensor->initializer != NULL) {
    tensor->initializer->init(this, tensor);
  }
}

void FFModel::map_weight(ParallelTensor weight, Op const *op) {
  switch (weight->num_dims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    map_weight_with_dim<DIM>(weight, op);                                      \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      // Unsupported dim
      assert(false);
    }
  }
}

template <int NDIM>
void FFModel::map_weight_with_dim(ParallelTensor weight,
                                  Op const *parallel_op) {
  // Step 0: check we are the owner or the owner is NULL
  // in which case set the owner to us
  if (weight->owner_op == NULL) {
    weight->owner_op = parallel_op;
    weight->owner_idx = -1; // meaning tensor is not an output of op
  } else {
    assert(weight->owner_op == parallel_op);
  }
  assert(parallel_op != NULL);
  int tdim = parallel_op->outputs[0]->num_dims;
  switch (parallel_op->op_type) {
    case OP_LINEAR:
    case OP_EMBEDDING:
    case OP_MULTIHEAD_ATTENTION: {
      switch (tdim) {
#define DIMFUNC(TDIM)                                                          \
  case TDIM: {                                                                 \
    map_linear_weight<NDIM, TDIM>(weight, parallel_op);                        \
    break;                                                                     \
  }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default: {
          assert(false);
        }
      }
      break;
    }
    case OP_CONV2D:
    case OP_BATCHNORM: {
      map_conv_weight<NDIM>(weight, parallel_op);
      break;
    }
    default: {
      fprintf(stderr,
              "FlexFlow currently does not support this weight"
              "type (%d). Report the error to the FlexFlow team.\n",
              parallel_op->op_type);
      assert(false && "Unsupported type for mapping weight");
    }
  }
}

bool FFModel::get_parallel_tensor_from_tensor(
    const Tensor tensor, ParallelTensor &parallel_tensor) const {
  // check if tensor->parallel_tensor is already set
  if (tensor->parallel_tensor != nullptr) {
    parallel_tensor = tensor->parallel_tensor;
    return true;
  }
  if (tensor->owner_layer != nullptr) {
    Op *mapped_op = nullptr;
    if (tensor->owner_layer->op_type == OP_INPUT) {
      // We use tensor_guid to match input operators
      size_t tensor_guid = tensor->owner_layer->outputs[0]->tensor_guid;
      for (auto const &op : operators) {
        if (op->op_type == OP_INPUT) {
          if (tensor_guid == ((NoOp *)op)->input_tensor_guid) {
            assert(mapped_op == nullptr);
            mapped_op = op;
          }
        }
      }
    } else {
      for (auto const &op : operators) {
        if (op->layer_guid == tensor->owner_layer->layer_guid) {
          assert(mapped_op == nullptr);
          mapped_op = op;
        }
      }
    }
    if (mapped_op != nullptr) {
      parallel_tensor = mapped_op->outputs[tensor->owner_idx];
      return true;
    }
  }
  assert(false);
  return true;
}

void FFModel::create_disjoint_partition(int num_dims,
                                        const ParallelDim dims[],
                                        IndexSpace const &part_is,
                                        LogicalRegion const &region,
                                        LogicalPartition &part) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, part_is);
  switch ((num_dims - 1) * MAX_TENSOR_DIM + task_domain.get_dim() - 1) {
#define DIMFUNC(NDIM, TDIM)                                                    \
  case (NDIM - 1) * MAX_TENSOR_DIM + (TDIM - 1): {                             \
    IndexSpaceT<TDIM> part_is_t(part_is);                                      \
    return create_disjoint_partition_with_dim2<NDIM, TDIM>(                    \
        dims, part_is_t, region, part);                                        \
  }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported NDIM/TDIM");
  }
}

template <int NDIM, int TDIM>
void FFModel::create_disjoint_partition_with_dim2(
    const ParallelDim dims[],
    IndexSpaceT<TDIM> const &part_is,
    LogicalRegion const &region,
    LogicalPartition &part) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, region.get_index_space());
  for (int i = 0; i < NDIM; i++) {
    int nparts = dims[i].degree;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      if (dims[i].parallel_idx == j) {
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      } else {
        transform[i][j] = 0;
      }
    }
  }
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part = runtime->get_logical_partition(ctx, region, ip);
}

void FFModel::create_aliased_partition(int num_dims,
                                       const ParallelDim dims[],
                                       int aliased_dim,
                                       IndexSpace const &part_is,
                                       LogicalRegion const &region,
                                       LogicalPartition &part) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, part_is);
  switch ((num_dims - 1) * MAX_TENSOR_DIM + task_domain.get_dim() - 1) {
#define DIMFUNC(NDIM, TDIM)                                                    \
  case (NDIM - 1) * MAX_TENSOR_DIM + (TDIM - 1): {                             \
    IndexSpaceT<TDIM> part_is_t(part_is);                                      \
    return create_aliased_partition_with_dim2<NDIM, TDIM>(                     \
        dims, aliased_dim, part_is_t, region, part);                           \
  }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported NDIM/TDIM");
  }
}

template <int NDIM, int TDIM>
void FFModel::create_aliased_partition_with_dim2(
    const ParallelDim dims[],
    int aliased_dim,
    IndexSpaceT<TDIM> const &part_is,
    LogicalRegion const &region,
    LogicalPartition &part) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, region.get_index_space());
  for (int i = 0; i < NDIM; i++) {
    int nparts = dims[i].degree;
    if (aliased_dim == i) {
      nparts = 1;
    }
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      if (dims[i].parallel_idx == j && i != aliased_dim) {
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      } else {
        transform[i][j] = 0;
      }
    }
  }
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, region.get_index_space(), part_is, transform, extent);
  // assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part = runtime->get_logical_partition(ctx, region, ip);
}

template <int NDIM>
void FFModel::create_disjoint_partition(const ParallelTensor tensor,
                                        IndexSpaceT<NDIM> const &part_is,
                                        LogicalPartition &part_fwd,
                                        LogicalPartition &part_bwd) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Check that dimension sizes match
  {
    assert(tensor->num_dims == NDIM);
    Domain domain = runtime->get_index_space_domain(ctx, part_is);
    assert(domain.get_dim() == NDIM);
  }
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, tensor->region.get_index_space());
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, NDIM> transform;
  Point<NDIM> ext_hi;
  for (int i = 0; i < NDIM; i++) {
    int nparts = part_rect.hi[i] - part_rect.lo[i] + 1;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < NDIM; j++) {
      if (i == j) {
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      } else {
        transform[i][j] = 0;
      }
    }
  }
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, tensor->region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor->region, ip);
  if (tensor->region_grad != LogicalRegion::NO_REGION) {
    // Current assume forward and grad share the same index space
    assert(tensor->region.get_index_space() ==
           tensor->region_grad.get_index_space());
    part_bwd = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
  } else {
    part_bwd = LogicalPartition::NO_PART;
  }
}

template <int NDIM, int TDIM>
void FFModel::create_data_parallel_partition_with_diff_dims(
    const ParallelTensor tensor,
    IndexSpaceT<TDIM> const &part_is,
    LogicalPartition &part_fwd,
    LogicalPartition &part_bwd) {
  assert(tensor->num_dims == NDIM);
  if (config.computationMode == COMP_MODE_TRAINING) {
    // Current assume forward and grad share the same index space
    if (tensor->region_grad != LogicalRegion::NO_REGION) {
      assert(tensor->region.get_index_space() ==
             tensor->region_grad.get_index_space());
    }
  }
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Rect<NDIM> rect =
      runtime->get_index_space_domain(ctx, tensor->region.get_index_space());
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  // Assume it is data parallel
  for (int i = 0; i < TDIM - 1; i++) {
    assert(part_rect.lo[i] == part_rect.hi[i]);
  }
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  for (int i = 0; i < NDIM; i++) {
    int nparts = 1;
    if (i == NDIM - 1) {
      nparts = part_rect.hi[TDIM - 1] - part_rect.lo[TDIM - 1] + 1;
    }
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      transform[i][j] = 0;
    }
  }
  transform[NDIM - 1][TDIM - 1] = extent.hi[NDIM - 1] - extent.lo[NDIM - 1] + 1;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, tensor->region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor->region, ip);
  if (config.computationMode == COMP_MODE_TRAINING) {
    if (tensor->region_grad != LogicalRegion::NO_REGION) {
      part_bwd = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
    }
  } else {
    part_bwd = LogicalPartition::NO_PART;
  }
}

// This function assumes:
// 1. the outer most dim of weight is channel out
// 2. partition is 2D (sample, channel_out)

template <int NDIM, int TDIM>
void FFModel::map_linear_weight(ParallelTensor weight, Op const *op) {
  assert(op->op_type == OP_LINEAR);
  std::string pcname = op->name;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, op->parallel_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++) {
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  }
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (weight->data_type) {
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int), FID_DATA);
      break;
    default:
      assert(false);
  }
  int out_channels = weight->dims[weight->num_dims - 1].size;
  // Step 1: forward region and partition
  if (weight->sync_type == ParameterSyncType::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    assert(out_channels % num_parts[0] == 0);
    hi[NDIM - 1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels / num_parts[0];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else if (weight->sync_type == ParameterSyncType::NCCL) {
    // FIXME: Currently only support the sample dimension for operators with
    // NCCL
    // for (int i = 0; i < TDIM-1; i++)
    //  assert(num_parts[i] == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++) {
      num_batches *= num_parts[i];
    }
    hi[NDIM - 1] = num_batches * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels / num_parts[0];
    for (int i = 1; i < TDIM; i++) {
      transform[NDIM - 1][i] = transform[NDIM - 1][i - 1] * num_parts[i - 1];
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else {
    assert(false);
  }
  // Step 2: initialize region
  if (weight->initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    weight->initializer->init(this, weight);
  }
  // Step 3: backward region
  if (weight->create_gradients &&
      config.computationMode == COMP_MODE_TRAINING) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++) {
      num_batches *= num_parts[i];
    }
    hi[NDIM - 1] = num_batches * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < TDIM; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels / num_parts[0];
    for (int i = 1; i < TDIM; i++) {
      transform[NDIM - 1][i] = transform[NDIM - 1][i - 1] * num_parts[i - 1];
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part_grad =
        runtime->get_logical_partition(ctx, weight->region_grad, ip);
  }
}

template <int NDIM>
void FFModel::map_conv_weight(ParallelTensor weight, Op const *op) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, op->parallel_is);
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  // Currently assume we do not split over the channel dimension
  assert(num_par_c == 1);
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (weight->data_type) {
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int), FID_DATA);
      break;
    default:
      assert(false);
  }
  // Step 1: forward region and partition
  int out_channels = weight->dims[weight->num_dims - 1].size;
  if (weight->sync_type == ParameterSyncType::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < 4; j++) {
        transform[i][j] = 0;
      }
    }
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, rect);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else if (weight->sync_type == ParameterSyncType::NCCL) {
    // Currently only support sample and attribute parallelism for NCCL
    // communication
    assert(num_par_c == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    hi[NDIM - 1] = num_par_n * num_par_h * num_par_w * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < 4; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels;
    transform[NDIM - 1][1] = out_channels * num_par_w;
    transform[NDIM - 1][2] = out_channels * num_par_w * num_par_h;
    transform[NDIM - 1][3] = out_channels * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part = runtime->get_logical_partition(ctx, weight->region, ip);
  } else {
    // Unsupported Parameter type
    assert(false);
  }
  // Step 2: initialize region
  if (weight->initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    weight->initializer->init(this, weight);
  }
  // Step 3: backward regin and partition
  if (weight->create_gradients &&
      config.computationMode == COMP_MODE_TRAINING) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++) {
      hi[i] = weight->dims[i].size - 1;
    }
    hi[NDIM - 1] = num_par_n * num_par_h * num_par_w * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM - 1] = out_channels - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < 4; j++) {
        transform[i][j] = 0;
      }
    }
    transform[NDIM - 1][0] = out_channels;
    transform[NDIM - 1][1] = out_channels * num_par_w;
    transform[NDIM - 1][2] = out_channels * num_par_w * num_par_h;
    transform[NDIM - 1][3] = out_channels * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, op->parallel_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part_grad =
        runtime->get_logical_partition(ctx, weight->region_grad, ip);
  }
}

template <int NDIM, int TDIM>
ParallelTensor FFModel::create_linear_replica(int const dims[],
                                              IndexSpaceT<TDIM> const &task_is,
                                              DataType data_type) {
  // No need to create replica for INFERENCE
  assert(config.computationMode == COMP_MODE_TRAINING);
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  assert(NDIM >= 2);
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++) {
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  }
  ParallelTensor replica = new ParallelTensorBase();
  replica->parallel_tensor_guid = parallel_tensor_global_guid++;
  replica->num_dims = NDIM;
  replica->data_type = data_type;
  for (int i = 0; i < NDIM; i++) {
    replica->dims[i].size = dims[NDIM - 1 - i];
  }
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (data_type) {
    case DT_FLOAT:
      allocator.allocate_field(sizeof(float), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(double), FID_DATA);
      break;
    case DT_INT32:
      allocator.allocate_field(sizeof(int), FID_DATA);
      break;
    default:
      assert(false);
  }
  Point<NDIM> hi;
  for (int i = 0; i < NDIM; i++) {
    hi[i] = dims[NDIM - 1 - i] - 1;
  }
  Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  replica->region_grad = runtime->create_logical_region(ctx, is, fs);
  assert(dims[0] == num_parts[0]);
  // assert(dims[1] % num_parts[1] == 0);
  hi[NDIM - 1] = dims[0] / num_parts[0] - 1;        // replication dim
  hi[NDIM - 2] = dims[1] / num_parts[TDIM - 1] - 1; // sample dim
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
  Transform<NDIM, TDIM> transform;
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < TDIM; j++) {
      transform[i][j] = 0;
    }
  }
  transform[NDIM - 1][0] = hi[NDIM - 1] + 1;
  transform[NDIM - 2][TDIM - 1] = hi[NDIM - 2] + 1;
  // transform[NDIM-2][1] = dims[1] / num_parts[1];
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  replica->part_grad =
      runtime->get_logical_partition(ctx, replica->region_grad, ip);
  return replica;
}

IndexSpace FFModel::get_task_is(MachineView const &view) const {
  auto const &iter = all_task_is.find(view);
  assert(iter != all_task_is.end());
  return iter->second;
}

IndexSpace FFModel::get_task_is(ParallelConfig const &pc) const {
  MachineView view;
  view.ndims = pc.nDims;
  for (int i = 0; i < view.ndims; i++) {
    view.dim[i] = pc.dim[i];
  }
  return get_task_is(view);
}

IndexSpace FFModel::get_or_create_task_is(const ParallelTensor tensor) {
  MachineView view;
  view.ndims = 0;
  for (int i = 0; i < tensor->num_dims; i++) {
    if (tensor->dims[i].parallel_idx >= 0) {
      view.dim[tensor->dims[i].parallel_idx] = tensor->dims[i].degree;
      view.ndims++;
    }
  }
  if (view.ndims == 0) {
    view.ndims = 1;
    view.dim[0] = 1;
  }
  return get_or_create_task_is(view);
}

IndexSpace FFModel::get_or_create_task_is(ParallelConfig const &pc) {
  MachineView view;
  view.ndims = pc.nDims;
  for (int i = 0; i < view.ndims; i++) {
    view.dim[i] = pc.dim[i];
  }
  return get_or_create_task_is(view);
}

IndexSpace FFModel::get_or_create_task_is(MachineView const &view) {
  if (all_task_is.find(view) != all_task_is.end()) {
    return all_task_is[view];
  }
  IndexSpace task_is;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  switch (view.ndims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> task_rect;                                                       \
    for (int i = 0; i < DIM; i++) {                                            \
      task_rect.lo[i] = 0;                                                     \
      task_rect.hi[i] = view.dim[i] - 1;                                       \
    }                                                                          \
    task_is = runtime->create_index_space(ctx, task_rect);                     \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  printf("ndim(%d) dims[%d %d %d %d]\n",
         view.ndims,
         view.dim[0],
         view.dim[1],
         view.dim[2],
         view.dim[3]);
  all_task_is[view] = task_is;
  return task_is;
}

IndexSpace FFModel::get_or_create_task_is(Domain const &domain) {
  MachineView view;
  view.ndims = domain.get_dim();
  for (int i = 0; i < view.ndims; i++) {
    view.dim[i] = domain.hi()[i] - domain.lo()[i] + 1;
  }
  return get_or_create_task_is(view);
}

/*
IndexSpace FFModel::get_or_create_task_is(int ndims, const std::string& pcname)
{
  ParallelConfig pc;
  bool result = config.find_parallel_config(ndims, pcname, pc);
  assert(result);
  return get_or_create_task_is(pc);
}

IndexSpace FFModel::get_task_is(int ndims, const std::string& pcname) const
{
  ParallelConfig pc;
  bool result = config.find_parallel_config(ndims, pcname, pc);
  assert(result);
  return get_task_is(pc);
}
*/

IndexSpace FFModel::get_task_is(Domain const &domain) const {
  MachineView view;
  view.ndims = domain.get_dim();
  for (int i = 0; i < view.ndims; i++) {
    view.dim[i] = domain.hi()[i] - domain.lo()[i] + 1;
  }
  auto const &iter = all_task_is.find(view);
  assert(iter != all_task_is.end());
  return iter->second;
}

void FFModel::reset_metrics() {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID,
                        TaskArgument(metrics_op, sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);
}

void FFModel::init_operators() {
  for (size_t i = 0; i < operators.size(); i++) {
    operators[i]->init(*this);
  }
}

void FFModel::forward(int seq_length) {
  iter_config.seq_length = seq_length;
  for (size_t i = 0; i < operators.size(); i++) {
    operators[i]->forward(*this);
  }
}

void FFModel::recompile_on_condition(RecompileState &r) {
  if (r.trigger()) {
    r.alter();
  }
}

void FFModel::compute_metrics() {
  Op *final_operator = get_final_operator();
  assert(final_operator->numOutputs == 1);
  metrics_op->compute(this, final_operator->outputs[0], parallel_label_tensor);
}

void FFModel::get_metrics() {
  metrics_input = operators.size() - 1;
}

void FFModel::backward(int seq_length) {
  iter_config.seq_length = seq_length;
  assert(config.computationMode == COMP_MODE_TRAINING);
  // Compute metrics
  compute_metrics();
  // Compute the gradients of the final operator wrt loss
  Op *final_operator = get_final_operator();
  assert(final_operator->numOutputs == 1);
  loss_op->backward(this, final_operator->outputs[0], parallel_label_tensor);
  // Perform backpropagation
  // std::set<LogicalRegion> resetedInputGrads;
  for (int l = operators.size() - 1; l >= 0; l--) {
#ifdef ENABLE_RESNET_INPUT_GRADIENT_OPTIMIZATION
    for (int i = 0; i < operators[l]->numInputs; i++) {
      if (resetedInputGrads.find(operators[l]->inputs[i]->region) ==
          resetedInputGrads.end()) {
        resetedInputGrads.insert(operators[l]->inputs[i]->region);
      } else {
        // This input's gradients has been reseted by other operators
        // So we should not do it again
        operators[l]->resetInputGrads[i] = false;
      }
    }
#endif
    // TODO: If operator serves for metrics and for further prop
    // if(l == metrics_input && metrics_input < (int)operators.size()-1)
    //  continue;
    operators[l]->backward(*this);
  }
}

void FFModel::update() {
  optimizer->next();
  for (size_t i = 0; i < parameters.size(); i++) {
    optimizer->update(parameters[i]);
  }
}

Op *FFModel::get_final_operator() const {
  int idx = operators.size() - 1;
  while (operators[idx]->op_type == OP_INPUT ||
         operators[idx]->op_type == OP_WEIGHT) {
    idx--;
  }
  // assert that the final operator has exactly one output
  assert(operators[idx]->numOutputs == 1);
  return operators[idx];
}

void FFModel::compile(Optimizer *_optimizer,
                      LossType loss_type,
                      std::vector<MetricsType> const &metrics,
                      CompMode comp_mode) {
  optimizer = _optimizer;
  compile(loss_type, metrics, comp_mode);
}

bool FFModel::apply_fusion(std::vector<Op *> const &operators,
                           std::vector<Op *> &new_operators) {
  // Context ctx = config.lg_ctx;
  // Runtime* runtime = config.lg_hlr;
  for (size_t l = 1; l < operators.size() - 1; l++) {
    // don't fuse input and weight operator since they don't involve any
    // forward/backward task launches
    if (operators[l]->op_type == OP_INPUT ||
        operators[l]->op_type == OP_WEIGHT) {
      continue;
    }
    // don't fuse parallel op since they have different parallel_is in
    // forward/backward
    if (operators[l]->is_parallel_op()) {
      continue;
    }
    size_t start = 0;
    {
      Op *opl = operators[l];
      for (int idx = 0; idx < opl->numInputs; idx++) {
        bool found = false;
        for (size_t i = 0; i < l; i++) {
          if (opl->inputs[idx]->owner_op == operators[i]) {
            assert(!found);
            found = true;
            if (i > start) {
              start = i;
            }
          }
        }
        assert(found || (opl->inputs[idx]->owner_op == NULL));
      }
    }
    for (size_t i = start; i < l; i++) {
      // Domain d1 =
      // runtime->get_index_space_domain(operators[l]->outputs[0]->parallel_is);
      // Domain d2 =
      // runtime->get_index_space_domain(operators[i]->outputs[0]->parallel_is);
      MachineView view1 = operators[l]->outputs[0]->machine_view;
      MachineView view2 = operators[i]->outputs[0]->machine_view;
      if (view1 == view2) {
        FusedOp *fused_op = nullptr;
        bool allocate_new_fused_op = false;
        if (operators[i]->op_type == OP_FUSED) {
          fused_op = (FusedOp *)operators[i];
        } else {
          //  cannot be an in-place operator
          if (operators[i]->has_inplace_output()) {
            continue;
          }
          // don't fuse input and weight operator since they don't involve any
          // forward/backward kernels
          if (operators[i]->op_type == OP_INPUT ||
              operators[i]->op_type == OP_WEIGHT) {
            continue;
          }
          // don't fuse parallel op since they have different parallel_is in
          // forward/backward
          if (operators[i]->is_parallel_op()) {
            continue;
          }
          fused_op = new FusedOp(*this, operators[i]);
          allocate_new_fused_op = true;
        }
        if (fused_op->add_operator(*this, operators[l])) {
          // Construct new operators
          new_operators.clear();
          for (size_t j = 0; j < i; j++) {
            new_operators.push_back(operators[j]);
          }
          new_operators.push_back(fused_op);
          for (size_t j = i + 1; j < operators.size(); j++) {
            if (j == l) {
              continue; // l and i are fused
            }
            Op *op = operators[j];
            // Update input tensors that belong to operator[l] or operator[i]
            for (int idx = 0; idx < op->numInputs; idx++) {
              if ((op->inputs[idx]->owner_op == operators[l]) ||
                  (op->inputs[idx]->owner_op == operators[i])) {
                int found = -1;
                for (int k = 0; k < fused_op->numOutputs; k++) {
                  if (fused_op->outputs[k]->region == op->inputs[idx]->region) {
                    assert(found == -1);
                    found = k;
                  }
                }
                assert(found >= 0);
                op->inputs[idx] = fused_op->outputs[found];
              }
            }
            // Insert op
            new_operators.push_back(op);
          }
          // We are exact one operator fewer than the original
          assert(new_operators.size() + 1 == operators.size());
          return true;
        } else {
          // TODO: delete fused_op to avoid memory leakage
          if (allocate_new_fused_op) {
            delete fused_op;
          }
          continue;
        }
      }
    }
  }
  return false;
}

Op *FFModel::create_operator_from_layer(
    Layer *layer, std::vector<ParallelTensor> const &inputs) {
  switch (layer->op_type) {
    case OP_INPUT: {
      // Input op cannot have an input
      assert(inputs.size() == 0);
      // Current assume we add one dimension before each tensor
      Tensor tensor = layer->outputs[0];
      int num_dims = tensor->num_dims;
      ParallelDim dims[MAX_TENSOR_DIM];
      for (int j = 0; j < num_dims; j++) {
        dims[j].size = tensor->dims[j];
        dims[j].degree = 1;
        dims[j].parallel_idx = -1;
        dims[j].is_replica_dim = false;
      }
      dims[num_dims].size = 1;
      dims[num_dims].degree = 1;
      dims[num_dims].parallel_idx = -1;
      dims[num_dims].is_replica_dim = true;
      // create_parallel_tensor adds an NoOp into operators
      ParallelTensor pt =
          create_parallel_tensor_legion_ordering(num_dims + 1,
                                                 dims,
                                                 tensor->data_type,
                                                 nullptr,
                                                 0,
                                                 true /*gradients*/,
                                                 tensor->tensor_guid);
      // assert that this tensor hasn't been mapped before
      assert(tensor->parallel_tensor == nullptr);
      tensor->parallel_tensor = pt;
      // start from data parllel tensor
      if (config.only_data_parallel) {
        Repartition *part = new Repartition(
            *this, pt, num_dims - 1, config.numNodes * config.workersPerNode);
        operators.push_back(part);
      }
      return operators[operators.size() - 1];
    }
    case OP_MULTIHEAD_ATTENTION: {
      Op *op =
          MultiHeadAttention::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_BATCHMATMUL: {
      Op *op = BatchMatmul::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_CAST: {
      Op *op = Cast::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_CONCAT: {
      Op *op = Concat::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_CONV2D: {
      Op *op = Conv2D::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_DROPOUT: {
      Op *op = Dropout::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_EMBEDDING: {
      Op *op = Embedding::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_EW_ADD:
    case OP_EW_SUB:
    case OP_EW_MUL:
    case OP_EW_DIV:
    case OP_EW_MAX:
    case OP_EW_MIN: {
      Op *op = ElementBinary::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_EXP:
    case OP_SIN:
    case OP_COS:
    case OP_SCALAR_MULTIPLY:
    case OP_SCALAR_ADD:
    case OP_SCALAR_SUB:
    case OP_SCALAR_TRUE_DIV:
    case OP_POW:
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    case OP_IDENTITY:
    case OP_GELU:
    case OP_ELU: {
      Op *op = ElementUnary::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_FLAT: {
      Op *op = Flat::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_GATHER: {
      Op *op = Gather::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_LAYERNORM: {
      Op *op = LayerNorm::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_LINEAR: {
      Op *op = Linear::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_POOL2D: {
      Op *op = Pool2D::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_REDUCE_SUM: {
      Op *op = Reduce::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_RESHAPE: {
      Op *op = Reshape::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_SOFTMAX: {
      Op *op = Softmax::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_SPLIT: {
      Op *op = Split::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_TRANSPOSE: {
      Op *op = Transpose::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_TOPK: {
      Op *op = TopK::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_GROUP_BY: {
      Op *op = Group_by::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_AGGREGATE: {
      Op *op = Aggregate::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    case OP_AGG_SPEC: {
      Op *op = Aggregate::create_operator_from_layer(*this, layer, inputs);
      operators.push_back(op);
      return op;
    }
    default:
      assert(false);
  }
}

void FFModel::create_operators_from_layers() {
  std::map<const Tensor, ParallelTensor> tensors_to_parallel_tensors;
  for (auto const &l : layers) {
    std::vector<ParallelTensor> inputs;
    for (int i = 0; i < l->numInputs; i++) {
      // create new input tensors
      assert(tensors_to_parallel_tensors.find(l->inputs[i]) !=
             tensors_to_parallel_tensors.end());
      inputs.push_back(tensors_to_parallel_tensors[l->inputs[i]]);
    }
    Op *op = create_operator_from_layer(l, inputs);
    assert(op->numOutputs == l->numOutputs);
    for (int i = 0; i < op->numOutputs; i++) {
      tensors_to_parallel_tensors[l->outputs[i]] = op->outputs[i];
    }
  }
}

void FFModel::compile(LossType loss_type,
                      std::vector<MetricsType> const &metrics,
                      CompMode comp_mode) {
  if (metrics_input == -1) {
    metrics_input = operators.size() - 1;
  }
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  config.computationMode = comp_mode;
  // if (config.import_strategy_file.length() > 0) {
  //   load_strategies_from_file(config.import_strategy_file,
  //   config.strategies);
  // }
  //  Construct operators from layers
  if (config.only_data_parallel) {
    fprintf(stderr,
            "Note: only_data_parallel is specified, FlexFlow compiles a "
            "data-parallel PCG.\n");
  }
  create_operators_from_layers();
  // Launch the graph optimize task
  {
    FFModel *model = this;
    TaskLauncher launcher(GRAPH_OPTIMIZE_TASK_ID,
                          TaskArgument(&model, sizeof(FFModel *)));
    Future future = runtime->execute_task(ctx, launcher);

    PCG::GraphOptimalViewSerialized ret =
        future.get_result<PCG::GraphOptimalViewSerialized>();
    Deserializer dez(ret.data, ret.total_bytes);
    // Reconstruct operators
    PCG::Graph *best_graph = new PCG::Graph(this);
    std::unordered_map<PCG::Node, MachineView> optimal_views;
    deserialize_graph_optimal_view(dez, best_graph, optimal_views);
    operators.clear();
    convert_graph_to_operators(best_graph, optimal_views);
    best_graph->print_dot();
    delete best_graph;
    for (auto const &layer : layers) {
      // map inputs to parallel tensor
      if (layer->op_type == OP_INPUT) {
        Tensor tensor = layer->outputs[0];
        ParallelTensor parallel_tensor = nullptr;
        for (auto const &op : operators) {
          if (op->op_type == OP_INPUT) {
            NoOp *noop = (NoOp *)op;
            if (noop->input_tensor_guid == tensor->tensor_guid) {
              parallel_tensor = op->outputs[0];
            }
          }
        }
        assert(parallel_tensor != nullptr);
        tensor->parallel_tensor = parallel_tensor;
      }
      // map weights to parallel_tensor
      for (int i = 0; i < layer->numWeights; i++) {
        assert(layer->weights[i] != nullptr);
        Tensor weight = layer->weights[i];
        ParallelTensor parallel_weight = nullptr;
        for (auto const &op : operators) {
          if (op->layer_guid == layer->layer_guid) {
            assert(op->op_type == layer->op_type);
            assert(op->numWeights == layer->numWeights);
            parallel_weight = op->weights[i];
          }
        }
        assert(parallel_weight != nullptr);
        weight->parallel_tensor = parallel_weight;
      }
    }
  }

  bool repl_labels = (operators[operators.size() - 1]->op_type == OP_AGG_SPEC);
  loss_op = new Loss(loss_type, repl_labels);
  metrics_op = new Metrics(loss_type, metrics);

  // Init performance metrics
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID,
                        TaskArgument(metrics_op, sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);

  // Perform inplace optimizations
  if (config.enable_inplace_optimizations) {
    for (size_t l = 1; l < operators.size(); l++) {
      if (operators[l]->can_inplace_output()) {
        // Assume outputs[0] is inplace with inputs[0]
        assert(operators[l]->numOutputs == 1);
        if (operators[l]->inputs[0]->owner_op != NULL) {
          // int dim1 = operators[l]->outputs[0]->num_dims;
          // int dim2 = operators[l]->inputs[0]->num_dims;
          MachineView view1 = operators[l]->outputs[0]->machine_view;
          MachineView view2 = operators[l]->inputs[0]->machine_view;
          if (view1 == view2) {
            // Check no others also need operators[l]->inputs[0]
            bool found = false;
            for (size_t i = 0; i < operators.size(); i++) {
              if (i == l) {
                continue;
              }
              for (int j = 0; j < operators[i]->numInputs; j++) {
                if ((operators[i]->inputs[j]->owner_op ==
                     operators[l]->inputs[0]->owner_op) &&
                    (operators[i]->inputs[j]->owner_idx ==
                     operators[l]->inputs[0]->owner_idx)) {
                  found = true;
                }
              }
            }
            if (!found) {
              // Perform inplace
              operators[l]->do_inplace_output();
            }
          }
        }
      }
    }
  }

  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->owner_op != NULL);
    }
    for (int i = 0; i < op->numWeights; i++) {
      assert(op->weights[i]->owner_op != NULL);
      assert(op->weights[i]->region != LogicalRegion::NO_REGION);
      parameters.push_back(op->weights[i]);
    }
    op->map_output_tensors(*this);
    // for (int i = 0; i < op->numOutputs; i++) {
    //   // Output tensor
    //   map_tensor(op->outputs[i], op);
    // }
    if (op->is_parallel_op()) {
      ((ParallelOp *)op)->create_input_partition(*this);
    }
    // op->map_output_tensors(*this);
  }

  // Check correctness
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i]->owner_op == op);
      assert(op->outputs[i]->owner_idx == i);
      assert(op->outputs[i]->parallel_tensor_guid != 0);
    }
  }

  // If an operator's input is training data
  // No need to compute its gradients
  for (size_t l = 0; l < operators.size(); l++) {
    Op *op = operators[l];
    for (int i = 0; i < op->numInputs; i++) {
      assert(op->inputs[i]->owner_op != nullptr);
      if (op->inputs[i]->owner_op->op_type == OP_INPUT) {
        op->trainableInputs[i] = false;
      }
    }
  }

  // Perform fusion optimizations
  if (config.perform_fusion) {
    fprintf(stderr, "Applying fusion optimizations during compilation...\n");
    fprintf(stderr, "%zu operators before fusion...\n", operators.size());
    std::vector<Op *> new_operators;
    std::vector<Op *> old_operators = operators;
    while (apply_fusion(operators, new_operators)) {
      for (size_t i = 0; i < new_operators.size(); i++) {
        for (int idx = 0; idx < new_operators[i]->numInputs; idx++) {
          for (size_t j = i + 1; j < new_operators.size(); j++) {
            if (new_operators[i]->inputs[idx]->owner_op == new_operators[j]) {
              assert(false);
            }
          }
        }
      }
      operators = new_operators;
    }
    // Check integrity
    for (size_t l = 0; l < operators.size(); l++) {
      if (operators[l]->op_type == OP_FUSED) {
        FusedOp *fused = (FusedOp *)operators[l];
        int ioff = 0, woff = 0, ooff = 0;
        for (int op = 0; op < fused->numOperators; op++) {
          Op *old_op = fused->operators[op];
          for (int i = 0; i < fused->op_num_inputs[op]; i++) {
            int my_off = fused->op_input_idx[i + ioff];
            if (fused->op_input_source[i + ioff] == FusedOp::SOURCE_INPUT) {
              assert(fused->inputs[my_off]->region ==
                     old_op->inputs[i]->region);
            } else if (fused->op_input_source[i + ioff] ==
                       FusedOp::SOURCE_OUTPUT) {
              assert(fused->outputs[my_off]->region ==
                     old_op->inputs[i]->region);
            } else {
              assert(false);
            }
          }
          for (int i = 0; i < fused->op_num_weights[op]; i++) {
            int my_off = fused->op_weight_idx[i + woff];
            assert(fused->op_weight_source[i + woff] == FusedOp::SOURCE_WEIGHT);
            assert(fused->weights[my_off]->region ==
                   old_op->weights[i]->region);
          }
          for (int i = 0; i < fused->op_num_outputs[op]; i++) {
            int my_off = fused->op_output_idx[i + ooff];
            assert(fused->op_output_source[i + ooff] == FusedOp::SOURCE_OUTPUT);
            assert(fused->outputs[my_off]->region ==
                   old_op->outputs[i]->region);
          }
          ioff += fused->op_num_inputs[op];
          woff += fused->op_num_weights[op];
          ooff += fused->op_num_outputs[op];
        }
      } else {
        bool found = false;
        for (size_t i = 0; i < old_operators.size(); i++) {
          if (old_operators[i] == operators[l]) {
            assert(!found);
            found = true;
          }
        }
        assert(found);
      }
    }
    fprintf(stderr, "%zu operators after fusion...\n", operators.size());
    for (size_t i = 0; i < operators.size(); i++) {
      Op *op = operators[i];
      printf("operator[%zu]: type(%s) guid(%lu)\n",
             i,
             get_operator_type_name(operators[i]->op_type).c_str(),
             operators[i]->op_guid);
      for (int j = 0; j < op->numInputs; j++) {
        LogicalRegion handle = op->inputs[j]->region;
        printf("inputs[%d] region(%d,%d,%d)\n",
               j,
               handle.get_index_space().get_id(),
               handle.get_field_space().get_id(),
               handle.get_tree_id());
      }
      for (int j = 0; j < op->numOutputs; j++) {
        LogicalRegion handle = op->outputs[j]->region;
        printf("outputs[%d] region(%d,%d,%d)\n",
               j,
               handle.get_index_space().get_id(),
               handle.get_field_space().get_id(),
               handle.get_tree_id());
      }
      for (int j = 0; j < op->numWeights; j++) {
        LogicalRegion handle = op->weights[j]->region;
        printf("weights[%d] region(%d,%d,%d)\n",
               j,
               handle.get_index_space().get_id(),
               handle.get_field_space().get_id(),
               handle.get_tree_id());
      }
    }
  }
  Op *final_operator = get_final_operator();
  // FIXME: currently assume the final operator has exactly one output
  assert(final_operator->numOutputs == 1);
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%d)\n", i, operators[i]->op_type);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("inputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("outputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
  // assert(final_operator->outputs[0].num_dims == 2);
  ParallelDim p_dims[MAX_TENSOR_DIM];
  int dims[MAX_TENSOR_DIM];
  int num_p_dims = final_operator->outputs[0]->num_dims;
  int num_dims = 0;
  // FIXME: Currently assume 1st input for 1st operator = batch_size
  for (int i = 0; i < num_p_dims; i++) {
    p_dims[i] = final_operator->outputs[0]->dims[i];
    if (!p_dims[i].is_replica_dim) {
      dims[num_dims++] = p_dims[i].size;
    }
  }
  DataType label_type = DT_FLOAT;
  if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    // assign dims[num_dims-1] = 1 for sparse categorical labels
    assert(p_dims[0].degree == 1);
    p_dims[0].size = 1;
    dims[0] = 1;
    label_type = DT_INT32;
  }
  // create label tensor
  switch (num_dims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    label_tensor = create_tensor_legion_ordering(                              \
        num_dims, dims, label_type, NULL, 0 /*idx*/, false /*create_grad*/);   \
    parallel_label_tensor = create_parallel_tensor_legion_ordering(            \
        num_p_dims, p_dims, label_type);                                       \
    label_tensor->parallel_tensor = parallel_label_tensor;                     \
    parallel_label_tensor->machine_view =                                      \
        final_operator->outputs[0]->machine_view;                              \
    map_tensor(parallel_label_tensor, parallel_label_tensor->owner_op);        \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      assert(false && "Unsupported dim");
    }
  }
  // init optimizer
  assert(optimizer != NULL);
  optimizer->init();

#ifdef FF_USE_NCCL
  if (config.computationMode == COMP_MODE_TRAINING) {
    // init all nccl communicators
    for (size_t l = 0; l < operators.size(); l++) {
      // Only create nccl for weights
      if (operators[l]->op_type != OP_WEIGHT) {
        continue;
      }
      MachineView view = operators[l]->outputs[0]->machine_view;
      if (view_hash_to_nccl_comms.find(view.hash()) ==
          view_hash_to_nccl_comms.end()) {
        TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
        Future future = runtime->execute_task(ctx, launcher);
        ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
        IndexSpace task_is = get_or_create_task_is(view);
        ArgumentMap argmap;
        IndexLauncher index_launcher(
            NCCL_INIT_COMMS_TASK_ID,
            task_is,
            TaskArgument(&ncclId, sizeof(ncclUniqueId)),
            argmap,
            Predicate::TRUE_PRED,
            false /*must*/,
            0 /*mapper_id*/,
            view.hash() /*MappingTagID*/);
        FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
        fm.wait_all_results();
        int idx = 0;
        Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
        ncclComm_t *nccl_comms =
            (ncclComm_t *)malloc(sizeof(ncclComm_t) * task_domain.get_volume());
        for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
          nccl_comms[idx] = fm.get_result<ncclComm_t>(*it);
        }
        view_hash_to_nccl_comms[view.hash()] = nccl_comms;
      }
    }
  }
#endif
}

struct PropagationEdgeInfo {
  Op *dstOp;
  size_t size;
};

float randf() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

#ifdef FF_USE_PROPAGATE
void FFModel::propagate(std::map<Op *, ParallelConfig> const &current,
                        std::map<Op *, ParallelConfig> &next) const {
  next = current;
  size_t opId = std::rand() % (operators.size() - 1);
  // TODO: need to make sure opId is not an output operator of the model
  assert(opId != operators.size() - 1);

  std::vector<PropagationEdgeInfo> choosable_edges;
  std::unordered_set<Op *> opsSeen;

  auto bwd_edge_map = this->get_bwd_edge_map();

  Op *selected_op = this->operators[opId];
  do {
    opsSeen.insert(selected_op);
    choosable_edges.clear();
    for (int i = 0; i < selected_op->numInputs; i++) {
      auto const &input = selected_op->inputs[i];
      if (opsSeen.find(input.owner_op) == opsSeen.end()) {
        PropagationEdgeInfo edgeInfo;
        edgeInfo.dstOp = selected_op->inputs[i].owner_op;
        if (edgeInfo.dstOp == NULL) {
          continue;
        }
        if (!edgeInfo.dstOp->is_adoptable_parallel_config(
                *this, next.at(selected_op))) {
          continue;
        }
        assert(edgeInfo.dstOp != NULL);
        edgeInfo.size = selected_op->inputs[i].get_volume();
        choosable_edges.push_back(edgeInfo);
      }
    }
    if (bwd_edge_map.find(selected_op) != bwd_edge_map.end()) {
      for (auto const &kv : bwd_edge_map.at(selected_op)) {
        if (opsSeen.find(kv.first) == opsSeen.end()) {
          PropagationEdgeInfo edgeInfo;
          edgeInfo.dstOp = kv.first;
          assert(edgeInfo.dstOp != NULL);
          if (!edgeInfo.dstOp->is_adoptable_parallel_config(
                  *this, next.at(selected_op))) {
            continue;
          }
          edgeInfo.size = kv.second;
          choosable_edges.push_back(edgeInfo);
        }
      }
    }

    if (choosable_edges.size() == 0) {
      break;
    }

    float avg_edge_size = 0.0f;
    for (auto const &edge : choosable_edges) {
      avg_edge_size += edge.size;
    }
    avg_edge_size /= choosable_edges.size();
    std::vector<float> edge_weights;
    for (auto const &edge : choosable_edges) {
      edge_weights.push_back(FFModel::PROPAGATION_SIZE_WEIGHT * edge.size +
                             avg_edge_size *
                                 (1 - FFModel::PROPAGATION_SIZE_WEIGHT));
    }
    assert(edge_weights.size() == choosable_edges.size());
    PropagationEdgeInfo chosenEdgeInfo =
        select_random(choosable_edges, edge_weights);

    auto const &dstOp = chosenEdgeInfo.dstOp;
    if (next.at(selected_op).is_data_parallel()) {
      next[dstOp] =
          next.at(selected_op)
              .change_data_parallel_dimensionality(dstOp->get_dimension());
      assert(dstOp->is_valid_parallel_config(*this, next.at(dstOp)));
    }
    selected_op = chosenEdgeInfo.dstOp;
  } while (randf() < FFModel::CONTINUE_PROPAGATION_CHANCE);
}
#endif

void FFModel::rewrite(std::map<Op const *, ParallelConfig> const &current,
                      std::map<Op const *, ParallelConfig> &next,
                      bool use_propagation) const {
  next = current;
  float propagate_chance;
  if (use_propagation) {
    propagate_chance = FFModel::PROPAGATION_CHANCE;
  } else {
    propagate_chance = 0.0f;
  }

  if (randf() < propagate_chance) {
#ifdef FF_USE_PROPAGATE
    this->propagate(current, next);
#endif
  } else {
    size_t opId = std::rand() % operators.size();
    // TODO: need to make sure opId is not an output operator of the model
    if (opId == operators.size() - 1) {
      return;
    }
    next[operators[opId]] = operators[opId]->get_random_parallel_config(*this);
  }
}

void FFModel::mcmc_optimize(std::map<Op const *, ParallelConfig> &best,
                            size_t budget,
                            float alpha,
                            CompMode comp_mode,
                            bool use_propagation) const {
  // Start from data parallel
  std::map<Op const *, ParallelConfig> current, next;
  float best_runtime = simulator->simulate_runtime(this, best, comp_mode);
  current = best;
  float current_runtime = best_runtime;
  size_t reset_span = budget / 100, last_reset_iter = 0;
  if (reset_span == 0) {
    reset_span = 1;
  }
  if (reset_span > 1000) {
    reset_span = 1000;
  }
  for (size_t iter = 0; iter <= budget; iter++) {
    // Reset the current strategy to be the best strategy
    if (iter - last_reset_iter >= reset_span) {
      current = best;
      current_runtime = best_runtime;
      last_reset_iter = iter;
    }
    rewrite(current, next, use_propagation);
    float next_runtime = simulator->simulate_runtime(this, next, comp_mode);
    if (iter % 1000 == 0) {
      printf("iteration(%zu) current_strategy(%.4lf) best_strategy(%.4lf)\n",
             iter,
             current_runtime,
             best_runtime);
    }
    float rn = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    // float ratio = (next_runtime - current_runtime) / current_runtime;
    float diff = (next_runtime - current_runtime);
    if (next_runtime < best_runtime) {
      best_runtime = next_runtime;
      best = next;
    }
    if (next_runtime < current_runtime) {
      current = next;
      current_runtime = next_runtime;
    } else if (rn < std::exp(-alpha * diff)) {
      current = next;
      current_runtime = next_runtime;
    }
  }
  printf("=========== Best Discovered Strategy ==========\n");
  simulator->simulate_runtime(
      this, best, comp_mode, this->config.export_strategy_task_graph_file);
  std::map<Op const *, ParallelConfig>::const_iterator it;
  for (it = best.begin(); it != best.end(); it++) {
    printf("[%s] num_dims(%d) dims[", it->first->name, it->second.nDims);
    for (int i = 0; i < it->second.nDims; i++) {
      if (i < it->second.nDims - 1) {
        printf("%d,", it->second.dim[i]);
      } else {
        printf("%d", it->second.dim[i]);
      }
    }
    printf("] device_ids[");
    for (int i = 0; i < it->second.num_parts(); i++) {
      if (i < it->second.num_parts() - 1) {
        printf("%d,", it->second.device_ids[i]);
      } else {
        printf("%d", it->second.device_ids[i]);
      }
    }
    printf("]\n");
  }
  printf("============= MCMC Search Finished ============\n\n");
}

void FFModel::zero_gradients(void) {
  for (int l = operators.size() - 1; l >= 0; l--) {
    operators[l]->zero_grad(*this);
  }
}

void FFModel::print_layers(int id) {
  if (id == -1) {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->print();
    }
  } else {
    layers[id]->print();
  }
}

std::unordered_map<Op *, std::vector<std::pair<Op *, int>>>
    FFModel::get_bwd_edge_map() const {
  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>> bwd_edge_map;
  for (auto const &op : this->operators) {
    for (int i = 0; i < op->numInputs; i++) {
      Op *src = (Op *)op->inputs[i]->owner_op;
      bwd_edge_map[src].push_back({op, op->inputs[i]->get_volume()});
    }
  }

  return bwd_edge_map;
};

PerfMetrics
    FFModel::update_metrics_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  Metrics *m = (Metrics *)task->args;
  if (task->futures.size() == 0) {
    // Create an empty future
    PerfMetrics perf;
    return perf;
  }
  assert(task->futures.size() > 1);
  PerfMetrics all_metrics = task->futures[0].get_result<PerfMetrics>();
  for (size_t i = 1; i < task->futures.size(); i++) {
    PerfMetrics one_metrics = task->futures[i].get_result<PerfMetrics>();
    all_metrics.update(one_metrics);
  }
  all_metrics.print(m);
  // fprintf(stderr, "acc_train_loss: %.4lf train_accuracy: %.2lf%%(%d/%d)\n",
  //         all_metrics.train_loss / all_metrics.train_all,
  //         all_metrics.train_correct * 100.0f / all_metrics.train_all,
  //         all_metrics.train_correct, all_metrics.train_all);
  return all_metrics;
}

// TODO: Move to an appropriate place
template <>
std::tuple<> get_input_shape(std::tuple<> const &) {
  return std::tuple<>();
}

template <>
std::tuple<ParallelTensorShape, ParallelTensorShape, ParallelTensorShape>
    get_input_shape(
        std::tuple<ParallelTensor, ParallelTensor, ParallelTensor> const
            &inputs) {
  return std::make_tuple(std::get<0>(inputs)->get_shape(),
                         std::get<1>(inputs)->get_shape(),
                         std::get<2>(inputs)->get_shape());
}

template <>
ParallelTensorShape get_input_shape(ParallelTensor const &input) {
  return input->get_shape();
}

template <>
std::pair<ParallelTensorShape, ParallelTensorShape>
    get_input_shape(std::pair<ParallelTensor, ParallelTensor> const &inputs) {
  return std::make_pair(inputs.first->get_shape(), inputs.second->get_shape());
}

template <>
std::vector<ParallelTensorShape>
    get_input_shape(std::vector<ParallelTensor> const &inputs) {
  std::vector<ParallelTensorShape> shapes;
  for (auto const &input : inputs) {
    shapes.push_back(input->get_shape());
  }
  return shapes;
}

void Op::prefetch(FFModel const &ff) {
  // TODO: perform prefetch for performance imporvement
}

// ========================================================
// class FFIterationConfig
// ========================================================
FFIterationConfig::FFIterationConfig() {
  seq_length = -1;
}

void FFIterationConfig::reset() {
  seq_length = -1;
}

// ========================================================
// class FFConfig
// ========================================================

// Default Config Parameters
struct DefaultConfig {
  const static int epochs = 1;
  // const static int iterations = 1;
  const static int batchSize = 64;
  const static bool profiling = false;
  constexpr static float learningRate = 0.01f;
  constexpr static float weightDecay = 0.0001f;
  const static size_t workSpaceSize = (size_t)1 * 1024 * 1024 * 1024; // 2GB
  const static int numNodes = 1;
  const static int workersPerNode = 0;
  const static int cpusPerNode = 0;
  const static size_t searchBudget = -1;
  const static size_t simulatorWorkSpaceSize =
      (size_t)2 * 1024 * 1024 * 1024; // 2GB
  constexpr static float searchAlpha = 1.2f;
  const static bool searchOverlapBackwardUpdate = false;
  const static bool onlyDataParallel = false;
  const static bool enableSampleParallel = true;
  const static bool enableParameterParallel = false;
  const static bool enableAttributeParallel = false;
  const static bool enableInplaceOptimizations = false;
  const static bool allowTensorOpMathConversion = false;
  const static int machine_model_version = 0;
  const static int simulator_segment_size = 16777216; // 16 MB
  const static int simulator_max_num_segments = 1;
  const static int base_optimize_threshold = 10;
  const static bool enable_control_replication = true;
  // The default python data loader type is 2 to enable control replication
  const static int python_data_loader_type = 2;
};

FFConfig::FFConfig() {
  epochs = DefaultConfig::epochs;
  // iterations = DefaultConfig::iterations;
  batchSize = DefaultConfig::batchSize;
  profiling = DefaultConfig::profiling;
  learningRate = DefaultConfig::learningRate;
  weightDecay = DefaultConfig::weightDecay;
  workSpaceSize = DefaultConfig::workSpaceSize;
  numNodes = DefaultConfig::numNodes;
  cpusPerNode = DefaultConfig::cpusPerNode;
  workersPerNode = DefaultConfig::workersPerNode;
  simulator_work_space_size = DefaultConfig::simulatorWorkSpaceSize;
  search_budget = DefaultConfig::searchBudget;
  search_alpha = DefaultConfig::searchAlpha;
  search_overlap_backward_update = DefaultConfig::searchOverlapBackwardUpdate;
  computationMode = COMP_MODE_TRAINING;
  only_data_parallel = DefaultConfig::onlyDataParallel;
  enable_sample_parallel = DefaultConfig::enableSampleParallel;
  enable_parameter_parallel = DefaultConfig::enableParameterParallel;
  enable_attribute_parallel = DefaultConfig::enableAttributeParallel;
  enable_inplace_optimizations = DefaultConfig::enableInplaceOptimizations;
  allow_tensor_op_math_conversion = DefaultConfig::allowTensorOpMathConversion;
  machine_model_version = DefaultConfig::machine_model_version;
  simulator_segment_size = DefaultConfig::simulator_segment_size;
  simulator_max_num_segments = DefaultConfig::simulator_max_num_segments;
  enable_control_replication = DefaultConfig::enable_control_replication;
  python_data_loader_type = DefaultConfig::python_data_loader_type;
  machine_model_file = "";
  import_strategy_file = "";
  export_strategy_file = "";
  export_strategy_task_graph_file = "";
  include_costs_dot_graph = false;
  export_strategy_computation_graph_file = "";
  dataset_path = "";
  substitution_json_path = tl::nullopt;
  syntheticInput = false;
  perform_fusion = false;
  base_optimize_threshold = DefaultConfig::base_optimize_threshold;
  perform_memory_search = false;

  // Parse input arguments
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_args(argv, argc);
  }
  // Use Real::Machine::get_address_space_count() to obtain the number of nodes
  numNodes = Realm::Machine::get_machine().get_address_space_count();

  Runtime *runtime = Runtime::get_runtime();
  lg_hlr = runtime;
  lg_ctx = Runtime::get_context();
  field_space = runtime->create_field_space(lg_ctx);
}

void FFConfig::parse_args(char **argv, int argc) {
  for (int i = 1; i < argc; i++) {
    if ((!strcmp(argv[i], "-e")) || (!strcmp(argv[i], "--epochs"))) {
      epochs = atoi(argv[++i]);
      continue;
    }
    // if ((!strcmp(argv[i], "-i")) || (!strcmp(argv[i], "--iterations"))) {
    //   iterations = atoi(argv[++i]);
    //   continue;
    // }
    if ((!strcmp(argv[i], "-b")) || (!strcmp(argv[i], "--batch-size"))) {
      batchSize = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--lr")) || (!strcmp(argv[i], "--learning-rate"))) {
      learningRate = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--wd")) || (!strcmp(argv[i], "--weight-decay"))) {
      weightDecay = atof(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-p")) || (!strcmp(argv[i], "--print-freq"))) {
      printFreq = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-d")) || (!strcmp(argv[i], "--dataset"))) {
      dataset_path = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--budget")) ||
        (!strcmp(argv[i], "--search-budget"))) {
      search_budget = (size_t)atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--alpha")) || (!strcmp(argv[i], "--search-alpha"))) {
      search_alpha = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-workspace-size")) {
      simulator_work_space_size = atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--import")) ||
        (!strcmp(argv[i], "--import-strategy"))) {
      import_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--export")) ||
        (!strcmp(argv[i], "--export-strategy"))) {
      export_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--only-data-parallel"))) {
      only_data_parallel = true;
      continue;
    }
    if ((!strcmp(argv[i], "--enable-parameter-parallel"))) {
      enable_parameter_parallel = true;
      continue;
    }
    if ((!strcmp(argv[i], "--enable-attribute-parallel"))) {
      enable_parameter_parallel = true;
      continue;
    }
    if (!strcmp(argv[i], "-ll:gpu")) {
      workersPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ll:fsize")) {
      device_mem = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--nodes")) {
      fprintf(stderr,
              "[Warning] --nodes is deprecated. "
              "FlexFlow will automatically detect the number of nodes.\n");
      numNodes = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ll:cpu")) {
      cpusPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--profiling")) {
      profiling = true;
      continue;
    }
    if (!strcmp(argv[i], "--allow-tensor-op-math-conversion")) {
      allow_tensor_op_math_conversion = true;
      continue;
    }
    if (!strcmp(argv[i], "--fusion")) {
      perform_fusion = true;
      continue;
    }
    if (!strcmp(argv[i], "--overlap")) {
      search_overlap_backward_update = true;
      continue;
    }
    if (!strcmp(argv[i], "--taskgraph")) {
      export_strategy_task_graph_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--include-costs-dot-graph")) {
      include_costs_dot_graph = true;
      continue;
    }
    if (!strcmp(argv[i], "--compgraph")) {
      export_strategy_computation_graph_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--machine-model-version")) {
      machine_model_version = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--machine-model-file")) {
      machine_model_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-segment-size")) {
      simulator_segment_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-max-num-segments")) {
      simulator_max_num_segments = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--enable-propagation")) {
      enable_propagation = true;
      continue;
    }
    if (!strcmp(argv[i], "--enable-inplace-optimizations")) {
      enable_inplace_optimizations = true;
      continue;
    }
    if (!strcmp(argv[i], "--search-num-nodes")) {
      search_num_nodes = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--search-num-workers")) {
      search_num_workers = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--base-optimize-threshold")) {
      base_optimize_threshold = atoi(argv[++i]);
    }
    if (!strcmp(argv[i], "--disable-control-replication")) {
      enable_control_replication = false;
      continue;
    }
    if (!strcmp(argv[i], "--python-data-loader-type")) {
      python_data_loader_type = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--substitution-json")) {
      substitution_json_path = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--memory-search")) {
      perform_memory_search = true;
      continue;
    }
  }
}

void register_flexflow_internal_tasks() {
  // CNN_INIT_TASK
  {
    TaskVariantRegistrar registrar(FF_INIT_TASK_ID, "cuda_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FFHandler, UtilityTasks::init_cuda_task>(
        registrar, "cuda_init_task");
  }
  // ElementUnary task
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_INIT_TASK_ID,
                                   "ElementWiseUnary Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, ElementUnary::init_task>(
        registrar, "ElementWiseUnary Init Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_FWD_TASK_ID,
                                   "ElementWiseUnary Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementUnary::forward_task>(
        registrar, "ElementWiseUnary Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_BWD_TASK_ID,
                                   "ElementWiseUnary Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementUnary::backward_task>(
        registrar, "ElementWiseUnary Backward Task");
  }
  // ElementBinary task
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_INIT_TASK_ID,
                                   "ElementWiseBinary Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, ElementBinary::init_task>(
        registrar, "ElementWiseBinary Init Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_FWD_TASK_ID,
                                   "ElementWiseBinary Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementBinary::forward_task>(
        registrar, "ElementWiseBinary Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_BWD_TASK_ID,
                                   "ElementWiseBinary Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementBinary::backward_task>(
        registrar, "ElementWiseBinary Backward Task");
  }
  // Cast
  {
    TaskVariantRegistrar registrar(CAST_INIT_TASK_ID, "Cast Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Cast::init_task>(
        registrar, "Cast Init Task");
  }
  {
    TaskVariantRegistrar registrar(CAST_FWD_TASK_ID, "Cast Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Cast::forward_task>(registrar,
                                                          "Cast Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CAST_BWD_TASK_ID, "Cast Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Cast::backward_task>(
        registrar, "Cast Backward Task");
  }
  // Conv2D task
  {
    TaskVariantRegistrar registrar(CONV2D_INIT_TASK_ID, "Conv2D Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Conv2D::init_task>(
        registrar, "Conv2D Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_FWD_TASK_ID, "Conv2D Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::forward_task>(
        registrar, "Conv2D Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CONV2D_BWD_TASK_ID, "Conv2D Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Conv2D::backward_task>(
        registrar, "Conv2D Backward Task");
  }
  //{
  //  TaskVariantRegistrar registrar(CONV2D_UPD_TASK_ID, "Conv2D Update");
  //  registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
  //  registrar.set_leaf();
  //  Runtime::preregister_task_variant<Conv2D::update_task>(
  //     registrar, "Conv2D Update Task");
  //}
  // Dropout task
  {
    TaskVariantRegistrar registrar(DROPOUT_INIT_TASK_ID, "Dropout Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Dropout::init_task>(
        registrar, "Dropout Init Task");
  }
  {
    TaskVariantRegistrar registrar(DROPOUT_FWD_TASK_ID, "Dropout Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Dropout::forward_task>(
        registrar, "Dropout Forward Task");
  }
  {
    TaskVariantRegistrar registrar(DROPOUT_BWD_TASK_ID, "Dropout Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Dropout::backward_task>(
        registrar, "Dropout Backward Task");
  }
  // Embedding task GPU
  {
    TaskVariantRegistrar registrar(EMBED_INIT_TASK_ID, "Embedding Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Embedding::init_task>(
        registrar, "Embedding Init Task");
  }
  {
    TaskVariantRegistrar registrar(EMBED_FWD_TASK_ID, "Embedding Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::forward_task>(
        registrar, "Embedding Forward Task");
  }
  {
    TaskVariantRegistrar registrar(EMBED_BWD_TASK_ID, "Embedding Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::backward_task>(
        registrar, "Embedding Backward Task");
  }
  // Embedding task CPU
  /* {
    TaskVariantRegistrar registrar(EMBED_FWD_TASK_ID, "Embedding Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::forward_task_cpu>(
        registrar, "Embedding Forward Task");
  }
  {
    TaskVariantRegistrar registrar(EMBED_BWD_TASK_ID, "Embedding Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Embedding::backward_task_cpu>(
        registrar, "Embedding Backward Task");
  }*/
  // Gather task
  {
    TaskVariantRegistrar registrar(GATHER_INIT_TASK_ID, "Gather Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Gather::init_task>(
        registrar, "Gather Init Task");
  }
  {
    TaskVariantRegistrar registrar(GATHER_FWD_TASK_ID, "Gather Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Gather::forward_task>(
        registrar, "Gather Forward Task");
  }
  {
    TaskVariantRegistrar registrar(GATHER_BWD_TASK_ID, "Gather Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Gather::backward_task>(
        registrar, "Gather Backward Task");
  }

  // Cache task CPU
  {
    TaskVariantRegistrar registrar(CACHE_INIT_TASK_ID, "Cache Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Cache::init_task>(
        registrar, "Cache Init Task");
  }
  {
    TaskVariantRegistrar registrar(CACHE_FWD_TASK_ID, "Cache Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Cache::forward_task>(
        registrar, "Cache Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CACHE_UPDATE_TASK_ID, "Cache Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<float, Cache::update_task>(
        registrar, "Cache Update Task");
  }
  // Group by task CPU
  {
    TaskVariantRegistrar registrar(GROUP_BY_INIT_TASK_ID, "Group_by Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Group_by::init_task>(
        registrar, "Group_by Init Task");
  }
  {
    TaskVariantRegistrar registrar(GROUP_BY_FWD_TASK_ID, "Group_by Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Group_by::forward_task>(
        registrar, "Group_by Forward Task");
  }
  {
    TaskVariantRegistrar registrar(GROUP_BY_BWD_TASK_ID, "Group_by Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Group_by::backward_task>(
        registrar, "Group_by Backward Task");
  }

  // Aggregate task CPU
  {
    TaskVariantRegistrar registrar(AGGREGATE_INIT_TASK_ID, "Aggregate Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Aggregate::init_task>(
        registrar, "Aggregate Init Task");
  }
  {
    TaskVariantRegistrar registrar(AGGREGATE_FWD_TASK_ID, "Aggregate Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Aggregate::forward_task>(
        registrar, "Aggregate Forward Task");
  }
  {
    TaskVariantRegistrar registrar(AGGREGATE_BWD_TASK_ID, "Aggregate Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Aggregate::backward_task>(
        registrar, "Aggregate Backward Task");
  }

  // AggregateSpec task CPU
  {
    TaskVariantRegistrar registrar(AGG_SPEC_INIT_TASK_ID,
                                   "Aggregate specification Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, AggregateSpec::init_task>(
        registrar, "Aggregate specification Init Task");
  }
  {
    TaskVariantRegistrar registrar(AGG_SPEC_FWD_TASK_ID,
                                   "Aggregate specification Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AggregateSpec::forward_task>(
        registrar, "Aggregate specification Forward Task");
  }
  {
    TaskVariantRegistrar registrar(AGG_SPEC_BWD_TASK_ID,
                                   "Aggregate specification Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AggregateSpec::backward_task>(
        registrar, "Aggregate specification Backward Task");
  }

  // Pool2D task
  {
    TaskVariantRegistrar registrar(POOL2D_INIT_TASK_ID, "pool2d_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Pool2D::init_task>(
        registrar, "pool2d_init_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_FWD_TASK_ID, "pool2d_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pool2D::forward_task>(registrar,
                                                            "pool2d_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_BWD_TASK_ID, "pool2d_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pool2D::backward_task>(registrar,
                                                             "pool2d_bwd_task");
  }
  // BatchNorm task
  {
    TaskVariantRegistrar registrar(BATCHNORM_INIT_TASK_ID, "bn_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, BatchNorm::init_task>(
        registrar, "bn_init_task");
  }
  {
    TaskVariantRegistrar registrar(BATCHNORM_FWD_TASK_ID, "bn_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::forward_task>(registrar,
                                                               "bn_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(BATCHNORM_BWD_TASK_ID, "bn_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::backward_task>(registrar,
                                                                "bn_bwd_task");
  }
  // BatchMatmul task
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_INIT_TASK_ID,
                                   "BatchMatmul Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, BatchMatmul::init_task>(
        registrar, "BatchMatmul Init Task");
  }
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_FWD_TASK_ID,
                                   "BatchMatmul Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchMatmul::forward_task>(
        registrar, "BatchMatmul Forward Task");
  }
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_BWD_TASK_ID,
                                   "BatchMatmul Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchMatmul::backward_task>(
        registrar, "BatchMatmul Backward Task");
  }
  // LayerNorm task
  {
    TaskVariantRegistrar registrar(LAYERNORM_INIT_TASK_ID,
                                   "layernorm_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, LayerNorm::init_task>(
        registrar, "layernorm_init_task");
  }
  {
    TaskVariantRegistrar registrar(LAYERNORM_FWD_TASK_ID, "layernorm_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LayerNorm::forward_task>(
        registrar, "layernorm_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(LAYERNORM_BWD_TASK_ID, "layernorm_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<LayerNorm::backward_task>(
        registrar, "layernorm_bwd_task");
  }
  // Linear task
  {
    TaskVariantRegistrar registrar(LINEAR_INIT_TASK_ID, "Linear Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Linear::init_task>(
        registrar, "Linear Init Task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_FWD_TASK_ID, "Linear Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::forward_task>(
        registrar, "Linear Forward Task");
  }
  {
    TaskVariantRegistrar registrar(LINEAR_BWD_TASK_ID, "Linear Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward_task>(
        registrar, "Linear Backward Task");
  }
  // Flat task
  {
    TaskVariantRegistrar registrar(FLAT_INIT_TASK_ID, "flat_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Flat::init_task>(
        registrar, "flat_init_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_FWD_TASK_ID, "flat_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::forward_task>(registrar,
                                                          "flat_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_BWD_TASK_ID, "flat_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::backward_task>(registrar,
                                                           "flat_bwd_task");
  }
  // Softmax task
  {
    TaskVariantRegistrar registrar(SOFTMAX_INIT_TASK_ID, "softmax_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Softmax::init_task>(
        registrar, "softmax_init_task");
  }
  {
    TaskVariantRegistrar registrar(SOFTMAX_FWD_TASK_ID, "softmax_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Softmax::forward_task>(
        registrar, "softmax_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(SOFTMAX_BWD_TASK_ID, "softmax_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Softmax::backward_task>(
        registrar, "softmax_bwd_task");
  }
  // compute Loss
  {
    TaskVariantRegistrar registrar(LOSS_BWD_TASK_ID, "Loss Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Loss::backward_task>(
        registrar, "Loss Backward Task");
  }
  // compute Metrics
  {
    TaskVariantRegistrar registrar(METRICS_COMP_TASK_ID, "Metrics Compute");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerfMetrics, Metrics::compute_task>(
        registrar, "Metrics Compute Task");
  }
  // MSELoss
  //{
  //  TaskVariantRegistrar registrar(MSELOSS_BWD_TASK_ID, "MSELoss Backward");
  //  registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
  //  registrar.set_leaf();
  //  Runtime::preregister_task_variant<PerfMetrics, MSELoss::backward_task>(
  //      registrar, "MSELoss Backward Task");
  //}
  // update metrics
  {
    TaskVariantRegistrar registrar(UPDATE_METRICS_TASK_ID, "Update Metrics");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerfMetrics,
                                      FFModel::update_metrics_task>(
        registrar, "Update Metrics Task");
  }
  // Concat task
  {
    TaskVariantRegistrar registrar(CONCAT_INIT_TASK_ID, "Concat Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Concat::init_task>(
        registrar, "Concat Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONCAT_FWD_TASK_ID, "Concat Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Concat::forward_task>(
        registrar, "Concat Forward Task");
  }
  {
    TaskVariantRegistrar registrar(CONCAT_BWD_TASK_ID, "Concat Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Concat::backward_task>(
        registrar, "Concat Backward Task");
  }
  // Split task
  {
    TaskVariantRegistrar registrar(SPLIT_INIT_TASK_ID, "Split Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Split::init_task>(
        registrar, "Split Init Task");
  }
  {
    TaskVariantRegistrar registrar(SPLIT_FWD_TASK_ID, "Split Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Split::forward_task>(
        registrar, "Split Forward Task");
  }
  {
    TaskVariantRegistrar registrar(SPLIT_BWD_TASK_ID, "Split Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Split::backward_task>(
        registrar, "Split Backward Task");
  }
  // Reduce task
  {
    TaskVariantRegistrar registrar(REDUCE_INIT_TASK_ID, "Reduce Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Reduce::init_task>(
        registrar, "Reduce Init Task");
  }
  {
    TaskVariantRegistrar registrar(REDUCE_FWD_TASK_ID, "Reduce Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduce::forward_task>(
        registrar, "Reduce Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REDUCE_BWD_TASK_ID, "Reduce Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduce::backward_task>(
        registrar, "Reduce Backward Task");
  }
  // Reshape task
  {
    TaskVariantRegistrar registrar(RESHAPE_INIT_TASK_ID, "Reshape Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Reshape::init_task>(
        registrar, "Reshape Init Task");
  }
  {
    TaskVariantRegistrar registrar(RESHAPE_FWD_TASK_ID, "Reshape Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reshape::forward_task>(
        registrar, "Reshape Forward Task");
  }
  {
    TaskVariantRegistrar registrar(RESHAPE_BWD_TASK_ID, "Reshape Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reshape::backward_task>(
        registrar, "Reshape Backward Task");
  }
  // Reverse task
  {
    TaskVariantRegistrar registrar(REVERSE_INIT_TASK_ID, "Reverse Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Reverse::init_task>(
        registrar, "Reverse Init Task");
  }
  {
    TaskVariantRegistrar registrar(REVERSE_FWD_TASK_ID, "Reverse Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reverse::forward_task>(
        registrar, "Reverse Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REVERSE_BWD_TASK_ID, "Reverse Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reverse::backward_task>(
        registrar, "Reverse Backward Task");
  }
  // Topk task
  {
    TaskVariantRegistrar registrar(TOPK_INIT_TASK_ID, "TopK Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, TopK::init_task>(
        registrar, "TopK Init Task");
  }
  {
    TaskVariantRegistrar registrar(TOPK_FWD_TASK_ID, "TopK Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TopK::forward_task>(registrar,
                                                          "TopK Forward Task");
  }
  {
    TaskVariantRegistrar registrar(TOPK_BWD_TASK_ID, "TopK Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TopK::backward_task>(
        registrar, "TopK Backward Task");
  }
  // Transpose task
  {
    TaskVariantRegistrar registrar(TRANSPOSE_INIT_TASK_ID, "Transpose Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Transpose::init_task>(
        registrar, "Transpose Init Task");
  }
  {
    TaskVariantRegistrar registrar(TRANSPOSE_FWD_TASK_ID, "Transpose Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Transpose::forward_task>(
        registrar, "Transpose Forward Task");
  }
  {
    TaskVariantRegistrar registrar(TRANSPOSE_BWD_TASK_ID, "Transpose Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Transpose::backward_task>(
        registrar, "Transpose Backward Task");
  }
  // MultiHeadAttention task
  {
    TaskVariantRegistrar registrar(ATTENTION_INIT_TASK_ID,
                                   "MultiHeadAttention Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, MultiHeadAttention::init_task>(
        registrar, "MultiHeadAttention Init Task");
  }
  {
    TaskVariantRegistrar registrar(ATTENTION_FWD_TASK_ID,
                                   "MultiHeadAttention Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<MultiHeadAttention::forward_task>(
        registrar, "MultiHeadAttention Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ATTENTION_BWD_TASK_ID,
                                   "MultiHeadAttention Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<MultiHeadAttention::backward_task>(
        registrar, "MultiHeadAttention Backward Task");
  }
  // NoOp
  {
    TaskVariantRegistrar registrar(NOOP_INIT_TASK_ID, "Weight NCCL Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, NoOp::init_task>(
        registrar, "Weight NCCL Init Task");
  }
  // FusedOp Task
  {
    TaskVariantRegistrar registrar(FUSEDOP_INIT_TASK_ID, "FusedOp Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, FusedOp::init_task>(
        registrar, "FusedOp Init Task");
  }
  {
    TaskVariantRegistrar registrar(FUSEDOP_FWD_TASK_ID, "FusedOp Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedOp::forward_task>(
        registrar, "FusedOp Forward Task");
  }
  {
    TaskVariantRegistrar registrar(FUSEDOP_BWD_TASK_ID, "FusedOp Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedOp::backward_task>(
        registrar, "FusedOp Backward Task");
  }
  // ParallelOp Task
  // Repartition
  {
    TaskVariantRegistrar registrar(REPARTITION_INIT_TASK_ID,
                                   "Repartition Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Repartition::init_task>(
        registrar, "Repartition init Task");
  }
  {
    TaskVariantRegistrar registrar(REPARTITION_FWD_TASK_ID,
                                   "Repartition Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Repartition::forward_task>(
        registrar, "Repartition Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REPARTITION_BWD_TASK_ID,
                                   "Repartition Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Repartition::backward_task>(
        registrar, "Repartition Backward Task");
  }
  // Combine
  {
    TaskVariantRegistrar registrar(COMBINE_INIT_TASK_ID, "Combine Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta *, Combine::init_task>(
        registrar, "Combine init Task");
  }
  {
    TaskVariantRegistrar registrar(COMBINE_FWD_TASK_ID, "Combine Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Combine::forward_task>(
        registrar, "Combine Forward Task");
  }
  {
    TaskVariantRegistrar registrar(COMBINE_BWD_TASK_ID, "Combine Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Combine::backward_task>(
        registrar, "Combine Backward Task");
  }
  // Replicate
  {
    TaskVariantRegistrar registrar(REPLICATE_FWD_TASK_ID, "Replicate Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Replicate::forward_task>(
        registrar, "Replicate Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REPLICATE_BWD_TASK_ID, "Replicate Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Replicate::backward_task>(
        registrar, "Replicate Backward Task");
  }
  // Reduction
  {
    TaskVariantRegistrar registrar(REDUCTION_FWD_TASK_ID, "Reduction Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduction::forward_task>(
        registrar, "Reduction Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REDUCTION_BWD_TASK_ID, "Reduction Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Reduction::backward_task>(
        registrar, "Reduction Backward Task");
  }
  // FusedParallelOp
  {
    TaskVariantRegistrar registrar(FUSED_PARALLELOP_FWD_TASK_ID,
                                   "FusedParallel Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedParallelOp::forward_task>(
        registrar, "FusedParallel Forward Task");
  }
  {
    TaskVariantRegistrar registrar(FUSED_PARALLELOP_BWD_TASK_ID,
                                   "FusedParallel Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedParallelOp::backward_task>(
        registrar, "FusedParallel Backward Task");
  }
  // Optimizer
  {
    TaskVariantRegistrar registrar(SGD_UPD_PS_TASK_ID,
                                   "SGD Parameter Server Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SGDOptimizer::ps_update_task>(
        registrar, "SGD Parameter Server Update Task");
  }
  {
    TaskVariantRegistrar registrar(ADAM_UPD_PS_TASK_ID,
                                   "Adam Parameter Server Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AdamOptimizer::ps_update_task>(
        registrar, "Adam Parameter Server Update Task");
  }
#ifdef FF_USE_NCCL
  {
    TaskVariantRegistrar registrar(SGD_UPD_NCCL_TASK_ID, "SGD NCCL Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SGDOptimizer::nccl_update_task>(
        registrar, "SGD NCCL Update Task");
  }
  {
    TaskVariantRegistrar registrar(ADAM_UPD_NCCL_TASK_ID, "Adam NCCL Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AdamOptimizer::nccl_update_task>(
        registrar, "Adam NCCL Update Task");
  }
#endif
  // Initializer
  {
    TaskVariantRegistrar registrar(ZERO_INIT_TASK_ID, "Zero Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ZeroInitializer::init_task_cpu>(
        registrar, "Zero Init Task");
  }
  {
    TaskVariantRegistrar registrar(ZERO_INIT_TASK_ID, "Zero Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ZeroInitializer::init_task>(
        registrar, "Zero Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONSTANT_INIT_TASK_ID, "Constant Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ConstantInitializer::init_task_cpu>(
        registrar, "Constant Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONSTANT_INIT_TASK_ID, "Constant Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ConstantInitializer::init_task>(
        registrar, "Constant Init Task");
  }
  {
    TaskVariantRegistrar registrar(UNIFORM_INIT_TASK_ID, "Uniform Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UniformInitializer::init_task>(
        registrar, "Uniform Init Task");
  }
  {
    TaskVariantRegistrar registrar(GLOROT_INIT_TASK_ID, "Glorot Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<GlorotUniform::init_task>(
        registrar, "Glorot Init Task");
  }
  {
    TaskVariantRegistrar registrar(NORMAL_INIT_TASK_ID, "Normalize Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<NormInitializer::init_task>(
        registrar, "Normalize Init Task");
  }
#ifdef FF_USE_NCCL
  // NCCL
  {
    TaskVariantRegistrar registrar(NCCL_GETUNIQUEID_TASK_ID,
                                   "NCCL GetUniqueId");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ncclUniqueId,
                                      Op::get_nccl_unique_id_task>(
        registrar, "NCCL GetUniqueId Task");
  }
  {
    TaskVariantRegistrar registrar(NCCL_INIT_COMMS_TASK_ID,
                                   "NCCL Init Communicators");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ncclComm_t, Op::init_nccl_comms_task>(
        registrar, "NCCL Init Communicators Task");
  }
#endif
  // Search
  {
    TaskVariantRegistrar registrar(STRATEGY_SEARCH_TASK_ID, "Stretegy Search");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Simulator::strategy_search_task>(
        registrar, "Stretegy Search Task");
  }
  // Graph optimize
  {
    TaskVariantRegistrar registrar(GRAPH_OPTIMIZE_TASK_ID, "Graph Optimize");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PCG::GraphOptimalViewSerialized,
                                      PCG::Graph::graph_optimize_task>(
        registrar, "Graph Optimize Task");
  }
  // Parameter Server Prefetch task
  {
    TaskVariantRegistrar registrar(PS_PREFETCH_TASK_ID, "Weights Prefetch");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UtilityTasks::dummy_task>(
        registrar, "Weights Prefetch Task");
  }
}

// template instantiations
#define DIMFUNC(DIM)                                                           \
  template Tensor FFModel::create_tensor<DIM>(int const dims[],                \
                                              DataType data_type,              \
                                              Layer const *owner_op,           \
                                              int owner_idx,                   \
                                              bool create_grad);               \
  template ParallelTensor FFModel::create_parallel_tensor<DIM>(                \
      const ParallelDim dims[],                                                \
      DataType data_type,                                                      \
      Op const *owner_op,                                                      \
      int owner_idx,                                                           \
      bool create_grad,                                                        \
      size_t input_tensor_guid);                                               \
  template ParallelParameter FFModel::create_parallel_weight<DIM>(             \
      const ParallelDim dims[],                                                \
      DataType data_type,                                                      \
      Op const *owner_op,                                                      \
      bool create_grad,                                                        \
      Initializer *initializer,                                                \
      ParameterSyncType sync_type);                                            \
  template void FFModel::map_tensor_with_dim<DIM>(ParallelTensor tensor,       \
                                                  Op const *parallel_op);      \
  template void FFModel::map_weight_with_dim<DIM>(ParallelTensor weight,       \
                                                  Op const *parallel_op);      \
  template Tensor FFModel::create_constant<DIM>(                               \
      int const *dims, float value, DataType data_type);                       \
  template void FFModel::create_disjoint_partition<DIM>(                       \
      const ParallelTensor tensor,                                             \
      IndexSpaceT<DIM> const &part_is,                                         \
      LogicalPartition &part_fwd,                                              \
      LogicalPartition &part_bwd);
LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC

#define DIMFUNC(D1, D2)                                                        \
  template void FFModel::map_tensor_with_dim2<D1, D2>(ParallelTensor tensor,   \
                                                      Op const *parallel_op);  \
  template void FFModel::create_disjoint_partition_with_dim2<D1, D2>(          \
      const ParallelDim dims[],                                                \
      IndexSpaceT<D2> const &part_is,                                          \
      LogicalRegion const &region,                                             \
      LogicalPartition &part);                                                 \
  template void FFModel::create_aliased_partition_with_dim2<D1, D2>(           \
      const ParallelDim dims[],                                                \
      int aliased_dim,                                                         \
      IndexSpaceT<D2> const &part_is,                                          \
      LogicalRegion const &region,                                             \
      LogicalPartition &part);                                                 \
  template void                                                                \
      FFModel::create_data_parallel_partition_with_diff_dims<D1, D2>(          \
          const ParallelTensor tensor,                                         \
          IndexSpaceT<D2> const &part_is,                                      \
          LogicalPartition &part_fwd,                                          \
          LogicalPartition &part_bwd);
LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

template void FFModel::map_conv_weight<4>(ParallelTensor weight,
                                          Op const *parallel_op);
template void FFModel::map_conv_weight<1>(ParallelTensor weight,
                                          Op const *parallel_op);

#define DIMFUNC(D1, D2)                                                        \
  template void FFModel::map_linear_weight<D1, D2>(ParallelTensor p,           \
                                                   Op const *op);
LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

#define DIMFUNC(D1, D2)                                                        \
  template ParallelTensor FFModel::create_linear_replica<D1>(                  \
      int const *dims, IndexSpaceT<D2> const &part_is, DataType data_type);
LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

}; // namespace FlexFlow
