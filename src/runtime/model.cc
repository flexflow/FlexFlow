/* Copyright 2019 Stanford
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
#include "model.h"
#include "cuda_helper.h"
#include "mapper.h"
#include "test_utils.h"
#include "dirent.h"
#include <unordered_set>
#include "random_utils.h"

using namespace std;
using namespace Legion;

LegionRuntime::Logger::Category log_model("Model");

TensorBase::TensorBase(void)
{
  ts_guid = 0;
  num_dims = 0;
  parallel_is = IndexSpace::NO_SPACE;
  region = LogicalRegion::NO_REGION;
  region_grad = LogicalRegion::NO_REGION;
  part = LogicalPartition::NO_PART;
  part_grad = LogicalPartition::NO_PART;
  owner_op = NULL;
  owner_idx = 0;
  data_type = DataType::DT_NONE;
  sync_type = ParameterSyncType::NONE;
  initializer = NULL;
  create_gradients = false;

  //physical_region.impl = NULL;
}

/*
Tensor& Tensor::operator=(const Tensor& rhs)
{
  guid = rhs.guid;
  num_dims = rhs.num_dims;
  for (int i = 0; i < num_dims; i++)
    dims[i].size = rhs.dims[i].size;
  data_type = rhs.data_type;
  sync_type = rhs.sync_type;
  initializer = rhs.initializer;
  owner_op = rhs->owner_op;
  owner_idx = rhs->owner_idx;
  create_gradients = rhs.create_gradients;
  region = rhs->region;
  region_grad = rhs->region_grad;
  part = rhs->part;
  part_grad = rhs->part_grad;
  physical_region = rhs.physical_region;
  return *this;
}

bool Tensor::operator==(const Tensor &rhs) const
{
  // We use guid to examine tensor equivalence
  return guid == rhs.guid;
}
*/

void TensorBase::inline_map(FFConfig &config)
{
  printf("inline map tensor\n");
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;

  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  InlineLauncher inline_launcher(region_req);
  physical_region = runtime->map_region(ctx, inline_launcher);
  physical_region.wait_until_valid();
}

void TensorBase::inline_unmap(FFConfig &config)
{
  printf("inline unmap tensor\n");
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  assert(physical_region.is_valid() == true);
  runtime->unmap_region(ctx, physical_region);
}

template<typename T>
T* TensorBase::get_raw_ptr(FFConfig &config)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  T *raw_ptr = NULL;
  if (num_dims == 1) {
    TensorAccessorW<T, 1> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (num_dims == 2) {
    TensorAccessorW<T, 2> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (num_dims == 3) {
    TensorAccessorW<T, 3> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (num_dims == 4) {
    TensorAccessorW<T, 4> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else {
    printf("wrong num_dims %d", num_dims);
    assert(0);
  }
  return raw_ptr;
}

void TensorBase::attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  AttachLauncher launcher(EXTERNAL_INSTANCE, region, region);
  std::vector<FieldID> fields(1, FID_DATA);
  const Memory local_sysmem = Machine::MemoryQuery(Machine::get_machine())
       .has_affinity_to(runtime->get_executing_processor(ctx))
       .only_kind(Memory::SYSTEM_MEM)
       .first();
  launcher.attach_array_soa(raw_ptr, column_major,
                            fields, local_sysmem);
  physical_region = runtime->attach_external_resource(ctx, launcher);
}

void TensorBase::detach_raw_ptr(FFConfig &config)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  runtime->detach_external_resource(ctx, physical_region);
}

bool TensorBase::get_input_sub_tensor(
    const ParallelConfig& pc,
    TensorBase& tensor,
    OperatorType type)
{
  //TODO: consider reduction dim for conv2d and linear
  switch (type) {
    case OP_FLAT:
      {
        assert (pc.nDims == 2 && "Invalid dimension for parallel config of OP_FLAT");
        int nonBatchDim = pc.dim[0];
        int batchDim = pc.dim[1];
        tensor.num_dims = num_dims;
        assert (nonBatchDim == 1 && "I'm not sure this is correct otherwise");
        if (dims[num_dims-1].size % batchDim != 0) {
          printf("Could not get input subtensor because the dimension is not divisiable: %d %% %d != 0\n", dims[num_dims-1].size, batchDim);
        }
        for (int i = num_dims - 2; i >= 0; i--) {
          tensor.dims[i].size = dims[i].size;
        }
        tensor.dims[num_dims-1].size = dims[num_dims-1].size / batchDim;
        break;
      }
    case OP_RESHAPE:
      {
        for (int i = 0; i < pc.nDims - 1; i ++)
          assert(pc.dim[i] == 1 && "Assuming data parallel for RESHAPE");
        int batchDim = pc.dim[pc.nDims-1];
        if (dims[num_dims-1].size % batchDim != 0) {
          printf("Could not get input subtensor because the dimension is not divisiable: %d %% %d != 0\n", dims[num_dims-1].size, batchDim);
        }
        tensor.num_dims = num_dims;
        for (int i = num_dims-2; i >= 0; i--) {
          tensor.dims[i].size = dims[i].size;
        }
        tensor.dims[num_dims-1].size = dims[num_dims-1].size / batchDim;
        break;
      }
    case OP_LINEAR:
    case OP_CONV2D:
      {
        if (pc.nDims != num_dims) {
          printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
          return false;
        }
        tensor.num_dims = num_dims;
        for (int i = 1; i < num_dims; i++) {
          if (dims[i].size % pc.dim[i] != 0) {
            printf("Could not get input subtensor because the given dimension is not divisible: %d %% %d != 0\n", dims[i].size, pc.dim[i]);
            return false;
          }
          tensor.dims[i].size = dims[i].size / pc.dim[i];
        }
        tensor.dims[0].size = dims[0].size;
        tensor.data_type = data_type;
	break;
      }
    default:
      {
        if (pc.nDims != num_dims) {
          printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
          return false;
        }
        for (int i = 0; i < num_dims; i++) {
          if (dims[i].size % pc.dim[i] != 0) {
            printf("Could not get input subtensor because the given dimension is not divisible: %d %% %d != 0\n", dims[i].size, pc.dim[i]);
            return false;
          }
        }
        tensor.num_dims = num_dims;
        for (int i = 0; i < num_dims; i++) {
          tensor.dims[i].size = dims[i].size / pc.dim[i];
        }
        tensor.data_type = data_type;
      }
      break;
  }
  return true;
}

bool TensorBase::get_output_sub_tensor(
    const ParallelConfig& pc,
    TensorBase& tensor,
    OperatorType type)
{
  if (pc.nDims != num_dims) {
    printf("Could not get output subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
    return false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].size % pc.dim[i] != 0) {
      printf("Could not get output subtensor because the given dimension is not divisible: %d %% %d != 0\n", dims[i].size, pc.dim[i]);
      return false;
    }
  }
  tensor.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++)
    tensor.dims[i].size = dims[i].size / pc.dim[i];
  tensor.data_type = data_type;
  return true;
}

size_t TensorBase::get_owner_independent_hash() const
{
  size_t hash = 17 * 31 + std::hash<int>()((int)data_type);
  hash = hash * 31 + std::hash<int>()((int)sync_type);
  hash = hash * 31 + std::hash<int>()(num_dims);
  for (int i = 0; i < num_dims; i++) {
    hash = hash * 31 + std::hash<int>()(dims[i].size);
    hash = hash * 31 + std::hash<int>()(dims[i].degree);
    hash = hash * 31 + std::hash<int>()(dims[i].parallel_idx);
  }
  return hash;
}

size_t TensorBase::get_volume() const
{
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++)
    volume *= dims[i].size;
  return volume;
}

size_t TensorBase::get_total_num_parts() const
{
  size_t parts = 1;
  for (int i = 0; i < num_dims; i++)
    parts *= dims[i].degree;
  return parts;
}

Domain TensorBase::get_domain() const
{
  Domain d;
  d.dim = this->num_dims;
  for (int i = 0; i < this->num_dims; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = this->dims[i].size - 1;
  }
  return d;
}

bool TensorBase::check_valid() const
{
  bool used[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM; i++)
    used[i] = false;
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].size % dims[i].degree != 0)
      return false;
    if (dims[i].parallel_idx > MAX_TENSOR_DIM)
      return false;
    if (dims[i].parallel_idx >= 0) {
      if (used[dims[i].parallel_idx])
        return false;
      used[dims[i].parallel_idx] = true;
    }
  }
  int idx = 0;
  while (used[idx]) idx++;
  for (int i = idx; i < MAX_TENSOR_DIM; i++)
    if (used[i]) return false;
  return true;
}

void TensorBase::print(const std::string& name) const
{
  printf("%s: sizes[", name.c_str());

  for (int i = 0; i < num_dims; i++) {
    printf("%d ", dims[i].size);
  }
  printf("] degree[");
  for (int i = 0; i < num_dims; i++)
    printf("%d ", dims[i].degree);
  printf("] parallel_ids[");
  for (int i = 0; i < num_dims; i++)
    printf("%d ", dims[i].parallel_idx);
  printf("]\n");

}

bool TensorBase::update_parallel_ids(
    int numdim,
    ParallelDim* dims)
{
  for (int i = 0; i < numdim; i++) {
    if (dims[i].degree == 1)
      dims[i].parallel_idx = -1;
  }
  bool used[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM; i++)
    used[i] = false;
  for (int i = 0; i < numdim; i++)
    if (dims[i].parallel_idx != -1) {
      assert(!used[dims[i].parallel_idx]);
      used[dims[i].parallel_idx] = true;
    }
  for (int i = 0; i < numdim; i++)
    if (dims[i].parallel_idx == -1 && dims[i].degree > 1) {
      int idx = 0;
      while (used[idx]) idx++;
      dims[i].parallel_idx = idx;
      used[idx] = true;
    }
  int idx = 0;
  while (used[idx]) idx++;
  for (int i = idx; i < MAX_TENSOR_DIM; i++)
    assert(!used[idx]);
  return true;
}

bool TensorBase::is_valid_machine_view(const MachineView& view) const
{
  int is_dim = 0;
  for (int i = 0; i < num_dims; i++)
    if (dims[i].parallel_idx != -1) {
      is_dim++;
      if (dims[i].parallel_idx > view.ndims)
        return false;
      if (view.dim[dims[i].parallel_idx] != dims[i].degree)
        return false;
    }
  if (is_dim == 0) {
    is_dim = 1;
  }
  if (is_dim != view.ndims)
    return false;
  if (get_total_num_parts() != view.num_parts())
    return false;
  return true;
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       int _numInputs,
       int _numWeights,
       const Tensor _input1,
       const Tensor _input2,
       const Tensor _input3,
       const Tensor _input4)
: op_type(_op_type), op_guid(model.op_global_guid++),
  numInputs(_numInputs), numWeights(_numWeights), numOutputs(1),
  profiling(model.config.profiling)
{
  for (int i = 0; i < MAX_NUM_INPUTS; i++)
    inputs[i] = NULL;
  std::vector<Tensor> tensors;
  tensors.push_back(_input1);
  tensors.push_back(_input2);
  tensors.push_back(_input3);
  tensors.push_back(_input4);
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(op_guid);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < numInputs + numWeights; i++) {
    assert(tensors[i] != NULL);
    if (i < numInputs) {
      // Activation
      inputs[i] = tensors[i];
    } else {
      // Weight
      weights[i - numInputs] = tensors[i];
    }
  }
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = NULL;
  }
  for (int i = 0; i < MAX_NUM_WORKERS; i++)
    meta[i] = NULL;
  parallel_dims_mapping = new std::vector<ParallelDimMappingRecord>();
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       int _numInputs,
       int _numWeights,
       const Tensor* _inputs)
: op_type(_op_type), op_guid(model.op_global_guid++),
  numInputs(_numInputs), numWeights(_numWeights), numOutputs(1),
  profiling(model.config.profiling)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
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
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i] = NULL;
  }
  for (int i = 0; i < MAX_NUM_WORKERS; i++)
    meta[i] = NULL;
  parallel_dims_mapping = new std::vector<ParallelDimMappingRecord>();
}

ParallelOp::ParallelOp(FFModel& model,
                       OperatorType op_type,
                       const char* name,
                       const Tensor input)
: Op(model, op_type, name, 1/*num_inputs*/, 0/*num_weights*/, input)
{}

bool Op::can_inplace_output()
{
  return false;
}

bool Op::has_inplace_output()
{
  return false;
}

void Op::do_inplace_output()
{
  assert(false);
}

Tensor Op::get_parameter(int index)
{
  assert(index < numWeights);
  return weights[index];
}

void Op::zero_grad(const FFModel& ff)
{
  // Do nothing for input and weight
  if (op_type == OP_INPUT || op_type == OP_WEIGHT)
    return;
  Runtime* runtime = ff.config.lg_hlr;
  Context ctx = ff.config.lg_ctx;
  ArgumentMap argmap;
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(ZERO_INIT_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(
        RegionRequirement(weights[i]->part_grad, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, weights[i]->region_grad));
    launcher.add_field(i, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
        RegionRequirement(outputs[i]->part_grad, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, outputs[i]->region_grad));
    //LogicalRegion lr = outputs[i]->region_grad;
    //printf("zero_grad:output[%d]: region(%d,%d,%d)\n", i, lr.get_index_space().get_id(), lr.get_field_space().get_id(), lr.get_tree_id());
    launcher.add_field(i + numWeights, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

ParallelConfig Op::get_data_parallel_config(const FFModel& ff) const
{
  int num_parts = ff.config.workersPerNode * ff.config.numNodes;
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = i;
  return pc;
}

void Op::create_input_partition(FFModel& model)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      create_input_partition_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      assert(false && "Unsupported dim");
    }
  }
}

template<int NDIM>
void Op::create_input_partition_with_dim(FFModel& model)
{
  std::string pcname = name;
  assert(numOutputs > 0);
  task_is = model.get_or_create_task_is(outputs[0]);
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  //Rect<NDIM> my_part_rect = runtime->get_index_space_domain(ctx, task_is);
  for (int i = 0; i < numInputs; i++) {
    input_lps[i] = inputs[i]->part;
    input_grad_lps[i] = inputs[i]->part_grad;
    switch (op_type) {
      case OP_REPARTITION:
      case OP_COMBINE:
      case OP_REPLICATE:
      case OP_REDUCTION:
      case OP_PIPELINE:
        break;
      default:
      {
        Domain my_domain = runtime->get_index_space_domain(ctx, task_is);
        Domain in_domain = runtime->get_index_space_domain(ctx, inputs[i]->parallel_is);
        assert(task_is == inputs[i]->parallel_is);
        assert(my_domain == in_domain);
      }
    }
#ifdef DEADCODE
    Rect<NDIM> input_part_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i]->part.get_index_partition());
    // sanity check
    // inputs and outputs should have the same ndim in the default case
    if (inputs[i]->owner_op != NULL) {
      std::string input_pcname = inputs[i]->owner_op->name;
      IndexSpaceT<NDIM> input_task_is;
      input_task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(
          NDIM, input_pcname));
      Rect<NDIM> input_part_rect2 = runtime->get_index_space_domain(
          ctx, input_task_is);
      assert(input_part_rect == input_part_rect2);
    }
    assert(my_part_rect == input_part_rect);
    if (my_part_rect == input_part_rect) {
      input_lps[i] = inputs[i]->part;
      input_grad_lps[i] = inputs[i]->part_grad;
    }
    else {
      // Assert that this input must be activations
      assert(inputs[i]->sync_type == ParameterSyncType::NONE);
      model.create_disjoint_partition(
          inputs[i], (IndexSpaceT<NDIM>)task_is,
          input_lps[i], input_grad_lps[i]);
    }
#endif
  }
}

ParallelConfig Op::get_random_parallel_config(const FFModel& ff) const
{
  std::vector<int> candidates;
  int batch_size = outputs[0]->dims[outputs[0]->num_dims-1].size;
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

Domain Op::get_output_tensor_shape(const ParallelConfig& pc,
    int output_idx, int part_idx) const
{
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

Domain Op::get_input_tensor_shape(const ParallelConfig& pc,
    int input_idx, int part_idx) const
{
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
    for (int i = 0; i < pc.nDims-1; i++)
      assert(pc.dim[i] == 1);
    for (int i = 0; i < d.dim-1; i++) {
      int dim_size = inputs[input_idx]->dims[i].size;
      d.rect_data[i] = 0;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    }
    // Assume an equal partitioning
    assert(inputs[input_idx]->dims[d.dim-1].size % pc.dim[pc.nDims-1] == 0);
    assert(part_idx < pc.dim[pc.nDims-1]);
    int dim_size = inputs[input_idx]->dims[d.dim-1].size / pc.dim[pc.nDims-1];
    d.rect_data[d.dim - 1] = part_idx * dim_size;
    d.rect_data[2*d.dim - 1] = d.rect_data[d.dim-1] + dim_size - 1;
    part_idx = part_idx / pc.dim[pc.nDims-1];
  }
  assert(part_idx == 0);
  return d;
}

Domain Op::get_weight_tensor_shape(const ParallelConfig& pc,
    int weight_idx, int part_idx) const
{
  // Default data parallel weight replication
  assert(weight_idx < numWeights);
  Domain d;
  d.dim = weights[weight_idx]->num_dims;
  for (int i = 0; i < d.dim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = weights[weight_idx]->dims[i].size - 1;
  }
  return d;
}

#ifdef FF_USE_NCCL
#ifdef DEADCODE
void Op::get_nccl_unique_id(const FFModel& ff)
{
  // Init NCCL id
  //int my_rank = -1, all_ranks = -1;
  //MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  //MPI_Comm_size(MPI_COMM_WORLD, &all_ranks);
  //if (my_rank == 0) ncclGetUniqueId(&ncclId);
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
  Future future = runtime->execute_task(ctx, launcher);
  ncclId = future.get_result<ncclUniqueId>();
  //MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  //fprintf(stderr, "In Op(%p): MPImyrank(%d) MPIallranks(%d) ncclID(%p)\n",
  //    this, my_rank, all_ranks, ncclId);
}
#endif

ncclUniqueId Op::get_nccl_unique_id_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  ncclUniqueId ncclId;
  checkNCCL(ncclGetUniqueId(&ncclId));
  return ncclId;
}

ncclComm_t Op::init_nccl_comms_task(const Task* task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime* runtime)
{
  // Must be an index space launch
  assert(task->is_index_space);
  ncclUniqueId ncclId = *((const ncclUniqueId*) task->args);
  int allRanks = task->index_domain.get_volume();
  assert(task->index_domain.contains(task->index_point));
  int myRank = 0;
  for (Domain::DomainPointIterator it(task->index_domain); it; it++, myRank++) {
    if (it.p == task->index_point) break;
  }
  ncclComm_t ncclComm;
  checkNCCL(ncclCommInitRank(&ncclComm, allRanks, ncclId, myRank));
  return ncclComm;
  //fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //    ncclComm, allRanks, myRank, ncclId);
}
#endif

ParallelDimMappingRecord::ParallelDimMappingRecord(void)
: output_dim(-1), input_dim(-1), weight_dim(-1),
  output_idx(-1), input_idx(-1), weight_idx(-1)
{}

void Op::register_output_input_parallel_dims(
    const Tensor output, int output_dim,
    const Tensor input, int input_dim)
{
  ParallelDimMappingRecord record;
  record.output_dim = output_dim;
  record.input_dim = input_dim;
  for (int i = 0; i < numOutputs; i++) {
    if (output == outputs[i])
      record.output_idx = i;
  }
  assert(record.output_idx >= 0);
  for (int i = 0; i < numInputs; i++) {
    if (input == inputs[i])
      record.input_idx = i;
  }
  assert(record.input_idx >= 0);
  parallel_dims_mapping->push_back(record);
}

void Op::register_output_weight_parallel_dims(
    const Tensor output, int output_dim,
    const Tensor weight, int weight_dim)
{
  ParallelDimMappingRecord record;
  record.output_dim = output_dim;
  record.weight_dim = weight_dim;
  for (int i = 0; i < numOutputs; i++) {
    if (output == outputs[i])
      record.output_idx = i;
  }
  assert(record.output_idx >= 0);
  for (int i = 0; i < numWeights; i++) {
    if (weight == weights[i])
      record.weight_idx = i;
  }
  assert(record.weight_idx >= 0);
  parallel_dims_mapping->push_back(record);
}

int Op::get_output_to_input_dim_mapping(
    const Tensor output, int output_dim,
    const Tensor input)
{
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
    if ((*parallel_dims_mapping)[i].output_idx != output_idx) continue;
    if ((*parallel_dims_mapping)[i].output_dim != output_dim) continue;
    if ((*parallel_dims_mapping)[i].input_idx != input_idx) continue;
    // Check validness
    assert((*parallel_dims_mapping)[i].weight_idx = -1);
    assert((*parallel_dims_mapping)[i].weight_dim = -1);
    return (*parallel_dims_mapping)[i].input_dim;
  }
  assert(false);
  return -1;
}

int Op::get_output_to_weight_dim_mapping(
    const Tensor output, int output_dim,
    const Tensor weight)
{
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
    if ((*parallel_dims_mapping)[i].output_idx != output_idx) continue;
    if ((*parallel_dims_mapping)[i].output_dim != output_dim) continue;
    if ((*parallel_dims_mapping)[i].weight_idx != weight_idx) continue;
    // Check validness
    assert((*parallel_dims_mapping)[i].input_idx = -1);
    assert((*parallel_dims_mapping)[i].input_dim = -1);
    return (*parallel_dims_mapping)[i].weight_dim;
  }
  assert(false);
  return -1;
}

bool Op::check_output_input_weight_parallel_dims()
{
  for (size_t i = 0; i < parallel_dims_mapping->size(); i++) {
    ParallelDimMappingRecord record = (*parallel_dims_mapping)[i];
    assert(record.input_idx == -1 || record.weight_idx == -1);
    int output_idx = record.output_idx;
    int output_dim = record.output_dim;
    if (record.weight_idx != -1) {
      int weight_idx = record.weight_idx;
      int weight_dim = record.weight_dim;
      assert(outputs[output_idx]->dims[output_dim].degree
          == weights[weight_idx]->dims[weight_dim].degree);
      assert(outputs[output_idx]->dims[output_dim].parallel_idx
          == weights[weight_idx]->dims[weight_dim].parallel_idx);
    } else if (record.input_idx != -1) {
      int input_idx = record.input_idx;
      int input_dim = record.input_dim;
      assert(outputs[output_idx]->dims[output_dim].degree
          == inputs[input_idx]->dims[input_dim].degree);
      assert(outputs[output_idx]->dims[output_dim].parallel_idx
          == inputs[input_idx]->dims[input_dim].parallel_idx);
    } else {
      assert(false);
    }
  }
  return true;
}

void Op::set_argumentmap_for_init(const FFModel& ff,
                                  ArgumentMap& argmap)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      ParallelConfig pc; \
      std::string pcname = name; \
      ff.config.find_parallel_config(DIM, pcname, pc); \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        FFHandler handle = ff.handlers[pc.device_ids[idx++]]; \
        handle.ncclComm = pc.nccl_comms[idx-1]; \
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void Op::set_opmeta_from_futuremap(const FFModel& ff,
                                   const FutureMap& fm)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        meta[idx++] = fm.get_result<OpMeta*>(*it); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void Op::set_argumentmap_for_forward(const FFModel& ff,
                                 ArgumentMap& argmap)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

void Op::set_argumentmap_for_backward(const FFModel& ff,
                                      ArgumentMap& argmap)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

bool Op::get_int_parameter(PMParameter para, int* value) const
{
  switch (para) {
    case PM_OP_TYPE:
      *value = (int) op_type;
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

bool Op::get_tensor_parameter(TNParameter tnp, DIMParameter dim, int* value) const
{
  if (tnp >= INPUT_0 && tnp <= INPUT_5)
    return get_input_parameter(tnp, dim, value);
  if (tnp >= WEIGHT_0 && tnp <= WEIGHT_5)
    return get_weight_parameter(tnp, dim, value);
  return false;
}

bool Op::get_input_parameter(TNParameter tnp, DIMParameter dim, int* value) const
{
  int inputIdx = 0, dimIdx = 0;
  assert(tnp <= INPUT_5 && tnp >= INPUT_0);
  inputIdx = tnp - INPUT_0;
  if (inputIdx >= numInputs) return false;
  switch (dim) {
    case DIM_3:
      dimIdx ++;
    case DIM_2:
      dimIdx ++;
    case DIM_1:
      dimIdx ++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = inputs[inputIdx]->num_dims;
      return true;
    default:
      return false;
  }
  if (dimIdx >= inputs[inputIdx]->num_dims) return false;
  *value = inputs[inputIdx]->dims[dimIdx].size;
  return true;
}

bool Op::get_weight_parameter(TNParameter tnp, DIMParameter dim, int* value) const
{
  int weightIdx = 0, dimIdx = 0;
  assert(tnp <= WEIGHT_5 && tnp >= WEIGHT_0);
  weightIdx = tnp - WEIGHT_0;
  if (weightIdx >= numWeights) return false;
  switch (dim) {
    case DIM_3:
      dimIdx ++;
    case DIM_2:
      dimIdx ++;
    case DIM_1:
      dimIdx ++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = weights[weightIdx]->num_dims;
      return true;
    default:
      return false;
  }
  if (dimIdx >= weights[weightIdx]->num_dims) return false;
  *value = weights[weightIdx]->dims[dimIdx].size;
  return true;
}

OpMeta::OpMeta(FFHandler _handle)
: handle(_handle)
{}

FFModel::FFModel(FFConfig& _config)
: op_global_guid(OP_GUID_FIRST_VALID),
  tensor_global_guid(TS_GUID_FIRST_VALID),
  node_global_guid(NODE_GUID_FIRST_VALID),
  config(_config),
  optimizer(NULL), loss_op(NULL), metrics_op(NULL), simulator(NULL)
{
  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
  // Register machine views
  register_machine_views();
  // Load strategy file
  int start_dim = 1, end_dim = 4;
#if MAX_TENSOR_DIM >= 5
  end_dim = 5;
#endif
  for (int i = start_dim; i <= end_dim; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::GPU;
    pc.nDims = i;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = config.workersPerNode * config.numNodes;
    for (int j = 0; j < pc.dim[pc.nDims-1]; j++)
      pc.device_ids[j] = j;
    config.strategies[FFConfig::DataParallelism_GPU_1D+i-1] = pc;
  }
  for (int i = start_dim; i <= end_dim; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::CPU;
    pc.nDims = i;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = config.cpusPerNode * config.numNodes;
    for (int j = 0; j < pc.dim[pc.nDims-1]; j++)
      pc.device_ids[j] = j;
    config.strategies[FFConfig::DataParallelism_CPU_1D+i-1] = pc;
  }

  // Create field space
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  // Build training dataset
  //if (config.datasetPath.length() == 0) {
  //  dataLoader = NULL;
  //} else {
  //  dataLoader = new DataLoader(config.datasetPath);
  //}

  ArgumentMap argmap;
  Rect<2> task_rect(Point<2>(0, 0),
                    Point<2>(0, config.workersPerNode * config.numNodes - 1));
  IndexSpaceT<2> task_is = runtime->create_index_space(ctx, task_rect);

  //int rank = 0;
  for (PointInRectIterator<2> it(task_rect); it(); it++) {
    FFInitInfo info;
    //info.myRank = rank++;
    //info.allRanks = config.workersPerNode * config.numNodes;
    info.workSpaceSize = config.workSpaceSize;
    info.allowTensorOpMathConversion = config.allow_tensor_op_math_conversion;
    argmap.set_point(*it, TaskArgument(&info, sizeof(FFInitInfo)));
  }

  // Init CUDA library on each worker
  IndexLauncher initLauncher(FF_INIT_TASK_ID, task_is,
                             TaskArgument(NULL, 0), argmap,
                             Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                             FFConfig::DataParallelism_GPU_2D);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<2> it(task_rect); it(); it++) {
    handlers[idx++] = fm.get_result<FFHandler>(*it);
  }
}

/*
template<int NDIM>
Tensor FFModel::create_tensor(const int dims[],
                              DataType data_type,
                              Op* owner_op,
                              bool create_grad)
{
  ParallelConfig pc;
  assert(config.find_parallel_config(NDIM, pc_name, pc));
  IndexSpaceT<NDIM> task_is = IndexSpaceT<NDIM>(get_or_create_task_is(pc));
  return create_tensor<NDIM>(dims, task_is, data_type, create_grad);
}
*/

template<int NDIM>
Tensor FFModel::create_constant(const int dims[],
                                float value,
                                DataType data_type)
{
  // FIXME: currently create gradients for constants since the current auto grad algorithm
  // computes gradients for all operators
  Tensor tensor = create_tensor<NDIM>(dims, data_type, NULL/*owner_op*/, true/*create_grad*/);
  IndexSpaceT<NDIM> part_is = (IndexSpaceT<NDIM>) get_or_create_task_is(NDIM, "");
  ConstantInitializer* init =  new ConstantInitializer(value);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  ArgumentMap argmap;
  IndexLauncher launcher(CONSTANT_INIT_TASK_ID, part_is,
      TaskArgument(init, sizeof(ConstantInitializer)), argmap,
      Predicate::TRUE_PRED, false, 0,
      FFConfig::get_hash_id(""));
  launcher.add_region_requirement(
      RegionRequirement(tensor->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, tensor->region));
  launcher.add_field(0, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return tensor;
}

Tensor FFModel::create_tensor(
    int numdim,
    const int dims[],
    DataType data_type,
    const Op* op,
    int idx,
    bool create_grad)
{
  switch (numdim) {
#define DIMFUNC(DIM) \
    case DIM: \
      return create_tensor<DIM>(dims, data_type, op, idx, create_grad);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported dim!");
  }
}

Tensor FFModel::create_tensor(
    int numdim,
    const ParallelDim dims[],
    DataType data_type,
    const Op* op,
    int idx,
    bool create_grad)
{
  switch (numdim) {
#define DIMFUNC(DIM) \
    case DIM: \
      return create_tensor<DIM>(dims, data_type, op, idx, create_grad);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported dim!");
  }
}

Tensor FFModel::create_tensor_legion_ordering(
    int numdim,
    const int dims[],
    DataType data_type,
    const Op* op,
    int idx,
    bool create_grad)
{
  int c_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    c_dims[i] = dims[numdim-1-i];
  return create_tensor(numdim, c_dims, data_type, op, idx, create_grad);
}

Tensor FFModel::create_tensor_legion_ordering(
    int numdim,
    const ParallelDim dims[],
    DataType data_type,
    const Op* op,
    int idx,
    bool create_grad)
{
  ParallelDim c_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    c_dims[i] = dims[numdim-1-i];
  return create_tensor(numdim, c_dims, data_type, op, idx, create_grad);
}

template<int NDIM>
Tensor FFModel::create_tensor(
    const int dims[],
    DataType data_type,
    const Op* owner_op,
    int owner_idx,
    bool create_grad)
{
  ParallelDim pdims[NDIM];
  for (int i = 0; i < NDIM; i++) {
    pdims[i].size = dims[i];
  }
  return create_tensor<NDIM>(pdims, data_type, owner_op, owner_idx, create_grad);
}

template<int NDIM>
Tensor FFModel::create_tensor(
    const ParallelDim dims[],
    DataType data_type,
    const Op* owner_op,
    int owner_idx,
    bool create_grad)
{
  Tensor tensor = new TensorBase();
  tensor->ts_guid = tensor_global_guid ++;
  tensor->data_type = data_type;
  if (owner_op == NULL) {
    NoOp* input_op = new NoOp(*this, OP_INPUT, tensor);
    layers.push_back(input_op);
    tensor->owner_op = input_op;
    tensor->owner_idx = 0;
  } else {
    tensor->owner_op = owner_op;
    tensor->owner_idx = owner_idx;
  }
  tensor->create_gradients = create_grad;
  tensor->num_dims = NDIM;
  for (int i = 0; i < NDIM; i++) {
    tensor->dims[i] = dims[NDIM-1-i];
  }
  assert(tensor->check_valid());
  return tensor;
}

template<int NDIM>
Parameter FFModel::create_weight(
    const int dims[],
    DataType data_type,
    const Op* owner_op,
    bool create_grad,
    Initializer* initializer,
    ParameterSyncType sync_type)
{
  ParallelDim pdims[NDIM];
  for (int i = 0; i < NDIM; i++)
    pdims[i].size = dims[i];
  return create_weight<NDIM>(pdims, data_type, owner_op,
      create_grad, initializer, sync_type);
}

template<int NDIM>
Parameter FFModel::create_weight(
    const ParallelDim dims[],
    DataType data_type,
    const Op* owner_op,
    bool create_grad,
    Initializer* initializer,
    ParameterSyncType sync_type)
{
  Parameter p = new TensorBase();
  p->ts_guid = tensor_global_guid ++;
  p->data_type = data_type;
  if (owner_op == NULL) {
    NoOp* weight_op = new NoOp(*this, OP_WEIGHT, p);
    layers.push_back(weight_op);
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
    p->dims[i] = dims[NDIM-1-i];
  }
  assert(p->get_volume() > 0);
  assert(p->check_valid());
  return p;
}

void FFModel::map_tensor(Tensor tensor, const Op* op)
{
  switch (tensor->num_dims) {
#define DIMFUNC(NDIM) \
    case NDIM: \
    { \
      map_tensor_with_dim<NDIM>(tensor, op); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim
      assert(false);
    }
  }
}

// Map tensor using parallelization strategies described in paralell_op
template<int NDIM>
void FFModel::map_tensor_with_dim(Tensor tensor, const Op* parallel_op)
{
  tensor->parallel_is = get_or_create_task_is(tensor);
  assert(tensor->owner_op != NULL);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, tensor->parallel_is);
  switch (task_domain.get_dim()) {
#define DIMFUNC(TDIM) \
    case TDIM: \
    { \
      map_tensor_with_dim2<NDIM, TDIM>(tensor, parallel_op); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      assert(false && "Unsupported Task Dim");
    }
  }
}

template<int NDIM, int TDIM>
void FFModel::map_tensor_with_dim2(Tensor tensor, const Op* parallel_op)
{
  // Step 0: check we are the owner or the owner is NULL
  // in which case set the owner to us
  if (tensor->owner_op == NULL) {
    tensor->owner_op = parallel_op;
    tensor->owner_idx = -1; // meaning tensor is not an output of op
  } else {
    assert(tensor->owner_op == parallel_op);
  }
  // Step 1: create regions
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;

  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator= runtime->create_field_allocator(ctx, fs);
  switch (tensor->data_type)
  {
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
  for (int i = 0; i < NDIM; i++)
    hi[i] = tensor->dims[i].size - 1;
  Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  tensor->region = runtime->create_logical_region(ctx, is, fs);
  if (tensor->create_gradients && config.computationMode == COMP_MODE_TRAINING) {
    tensor->region_grad = runtime->create_logical_region(ctx, is, fs);
  }
  // Step 2: create partitions if parallel_op != NULL
  if (parallel_op != NULL) {
    IndexSpaceT<TDIM> part_is = (IndexSpaceT<TDIM>) get_or_create_task_is(tensor);
    //Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
    Transform<NDIM, TDIM> transform;
    Point<NDIM> ext_hi;
    for (int i = 0; i < NDIM; i++) {
      int nparts = tensor->dims[i].degree;
      ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
    }
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        if (tensor->dims[i].parallel_idx == j)
          transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
        else
          transform[i][j] = 0;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
    tensor->part = runtime->get_logical_partition(ctx, tensor->region, ip);
    if (tensor->create_gradients && config.computationMode == COMP_MODE_TRAINING) {
      tensor->part_grad = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
    }
  }
  // Step 3: initialize the tensor
  if (tensor->initializer != NULL) {
    tensor->initializer->init(this, tensor);
  }
}

void FFModel::map_weight(Tensor weight, const Op* op)
{
  switch (weight->num_dims) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      map_weight_with_dim<DIM>(weight, op); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim
      assert(false);
    }
  }
}

template<int NDIM>
void FFModel::map_weight_with_dim(Tensor weight, const Op* parallel_op)
{
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
    case OP_MULTIHEAD_ATTENTION:
    {
      switch (tdim) {
#define DIMFUNC(TDIM) \
        case TDIM: \
        { \
          map_linear_weight<NDIM, TDIM>(weight, parallel_op); \
          break; \
        }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
        {
          assert(false);
        }
      }
      break;
    }
    case OP_CONV2D:
    case OP_BATCHNORM:
    {
      map_conv_weight<NDIM>(weight, parallel_op);
      break;
    }
    default:
    {
      fprintf(stderr, "FlexFlow currently does not support this weight"
          "type (%d). Report the error to the FlexFlow team.\n",
          parallel_op->op_type);
      assert(false && "Unsupported type for mapping weight");
    }
  }
}

void FFModel::create_disjoint_partition(int num_dims,
                                        const ParallelDim dims[],
                                        const IndexSpace& part_is,
                                        const LogicalRegion& region,
                                        LogicalPartition& part)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, part_is);
  switch ((num_dims-1)*MAX_TENSOR_DIM+task_domain.get_dim()-1) {
#define DIMFUNC(NDIM, TDIM) \
    case (NDIM-1)*MAX_TENSOR_DIM+(TDIM-1): \
    { \
      IndexSpaceT<TDIM> part_is_t(part_is); \
      return create_disjoint_partition_with_dim2<NDIM, TDIM>( \
          dims, part_is_t, region, part);  \
    }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported NDIM/TDIM");
  }
}

template<int NDIM, int TDIM>
void FFModel::create_disjoint_partition_with_dim2(const ParallelDim dims[],
                                                  const IndexSpaceT<TDIM>& part_is,
                                                  const LogicalRegion& region,
                                                  LogicalPartition& part)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  //Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, region.get_index_space());
  for (int i = 0; i < NDIM; i++) {
    int nparts = dims[i].degree;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < TDIM; j++)
      if (dims[i].parallel_idx == j)
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      else
        transform[i][j] = 0;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part = runtime->get_logical_partition(ctx, region, ip);
}

void FFModel::create_aliased_partition(int num_dims,
                                       const ParallelDim dims[],
                                       int aliased_dim,
                                       const IndexSpace& part_is,
                                       const LogicalRegion& region,
                                       LogicalPartition& part)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Domain task_domain = runtime->get_index_space_domain(ctx, part_is);
  switch ((num_dims-1)*MAX_TENSOR_DIM+task_domain.get_dim()-1) {
#define DIMFUNC(NDIM, TDIM) \
    case (NDIM-1)*MAX_TENSOR_DIM+(TDIM-1): \
    { \
      IndexSpaceT<TDIM> part_is_t(part_is); \
      return create_aliased_partition_with_dim2<NDIM, TDIM>( \
          dims, aliased_dim, part_is_t, region, part);  \
    }
    LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false && "Unsupported NDIM/TDIM");
  }
}

template<int NDIM, int TDIM>
void FFModel::create_aliased_partition_with_dim2(const ParallelDim dims[],
                                                 int aliased_dim,
                                                 const IndexSpaceT<TDIM>& part_is,
                                                 const LogicalRegion& region,
                                                 LogicalPartition& part)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  //Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, region.get_index_space());
  for (int i = 0; i < NDIM; i++) {
    int nparts = dims[i].degree;
    if (aliased_dim == i)
      nparts = 1;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < TDIM; j++)
      if (dims[i].parallel_idx == j && i != aliased_dim)
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      else
        transform[i][j] = 0;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, region.get_index_space(), part_is, transform, extent);
  //assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part = runtime->get_logical_partition(ctx, region, ip);
}

template<int NDIM>
void FFModel::create_disjoint_partition(const Tensor tensor,
                                        const IndexSpaceT<NDIM>& part_is,
                                        LogicalPartition& part_fwd,
                                        LogicalPartition& part_bwd)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  // Check that dimension sizes match
  {
    assert(tensor->num_dims == NDIM);
    Domain domain = runtime->get_index_space_domain(ctx, part_is);
    assert(domain.get_dim() == NDIM);
  }
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, tensor->region.get_index_space());
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, NDIM> transform;
  Point<NDIM> ext_hi;
  for (int i = 0; i < NDIM; i++) {
    int nparts = part_rect.hi[i] - part_rect.lo[i] + 1;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < NDIM; j++)
      if (i == j)
        transform[i][j] = extent.hi[i] - extent.lo[i] + 1;
      else
        transform[i][j] = 0;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, tensor->region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor->region, ip);
  if (tensor->region_grad != LogicalRegion::NO_REGION) {
    // Current assume forward and grad share the same index space
    assert(tensor->region.get_index_space() == tensor->region_grad.get_index_space());
    part_bwd = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
  } else {
    part_bwd = LogicalPartition::NO_PART;
  }
}

template<int NDIM, int TDIM>
void FFModel::create_data_parallel_partition_with_diff_dims(const Tensor tensor,
                                                            const IndexSpaceT<TDIM>& part_is,
                                                            LogicalPartition& part_fwd,
                                                            LogicalPartition& part_bwd)
{
  assert(tensor->num_dims == NDIM);
  if (config.computationMode == COMP_MODE_TRAINING) {
    // Current assume forward and grad share the same index space
    assert(tensor->region.get_index_space() == tensor->region_grad.get_index_space());
  }
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, tensor->region.get_index_space());
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  // Assume it is data parallel
  for (int i = 0; i < TDIM - 1; i++)
    assert(part_rect.lo[i] == part_rect.hi[i]);
  Transform<NDIM, TDIM> transform;
  Point<NDIM> ext_hi;
  for (int i = 0; i < NDIM; i++) {
    int nparts = 1;
    if (i == NDIM - 1)
      nparts = part_rect.hi[TDIM-1] - part_rect.lo[TDIM-1] + 1;
    ext_hi[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), ext_hi);
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < TDIM; j++)
      transform[i][j] = 0;
  transform[NDIM-1][TDIM-1] = extent.hi[NDIM-1] - extent.lo[NDIM-1] + 1;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, tensor->region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor->region, ip);
  if (config.computationMode == COMP_MODE_TRAINING) {
    part_bwd = runtime->get_logical_partition(ctx, tensor->region_grad, ip);
  } else {
    part_bwd = LogicalPartition::NO_PART;
  }
}

// This function assumes:
// 1. the outer most dim of weight is channel out
// 2. partition is 2D (sample, channel_out)

template<int NDIM, int TDIM>
void FFModel::map_linear_weight(
    Tensor weight,
    const Op* op)
{
  assert(op->op_type == OP_LINEAR);
  std::string pcname = op->name;
  IndexSpaceT<TDIM> part_is = (IndexSpaceT<TDIM>)get_or_create_task_is(TDIM, pcname);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++)
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator= runtime->create_field_allocator(ctx, fs);
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
  int out_channels = weight->dims[weight->num_dims-1].size;
  // Step 1: forward region and partition
  if (weight->sync_type == ParameterSyncType::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = weight->dims[i].size-1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    assert(out_channels % num_parts[0] == 0);
    hi[NDIM-1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = out_channels / num_parts[0];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight->part = runtime->get_logical_partition(
        ctx, weight->region, ip);
  } else if (weight->sync_type == ParameterSyncType::NCCL) {
    // FIXME: Currently only support the sample dimension for operators with NCCL
    //for (int i = 0; i < TDIM-1; i++)
    //  assert(num_parts[i] == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = weight->dims[i].size-1;
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++)
      num_batches *= num_parts[i];
    hi[NDIM-1] = num_batches * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = out_channels / num_parts[0];
    for (int i = 1; i < TDIM; i++)
      transform[NDIM-1][i] = transform[NDIM-1][i-1] * num_parts[i-1];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part = runtime->get_logical_partition(
        ctx, weight->region, ip);
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
  if (weight->create_gradients && config.computationMode == COMP_MODE_TRAINING) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = weight->dims[i].size-1;
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++)
      num_batches *= num_parts[i];
    hi[NDIM-1] = num_batches * out_channels -1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = out_channels / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = out_channels / num_parts[0];
    for (int i = 1; i < TDIM; i++)
      transform[NDIM-1][i] = transform[NDIM-1][i-1] * num_parts[i-1];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part_grad = runtime->get_logical_partition(
        ctx, weight->region_grad, ip);
  }
}

template<int NDIM>
void FFModel::map_conv_weight(
    Tensor weight,
    const Op* parallel_op)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  std::string pcname = parallel_op->name;
  IndexSpaceT<4> part_is = (IndexSpaceT<4>) get_or_create_task_is(4, pcname);
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, part_is);
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  // Currently assume we do not split over the channel dimension
  assert(num_par_c == 1);
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator= runtime->create_field_allocator(ctx, fs);
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
  int out_channels = weight->dims[weight->num_dims-1].size;
  if (weight->sync_type == ParameterSyncType::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = weight->dims[i].size-1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 4; j++)
        transform[i][j] = 0;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, rect);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight->part = runtime->get_logical_partition(
        ctx, weight->region, ip);
  } else if (weight->sync_type == ParameterSyncType::NCCL) {
    // Currently only support sample and attribute parallelism for NCCL communication
    assert(num_par_c == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = weight->dims[i].size-1;
    hi[NDIM-1] = num_par_n * num_par_h * num_par_w * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = out_channels-1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 4; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = out_channels;
    transform[NDIM-1][1] = out_channels * num_par_w;
    transform[NDIM-1][2] = out_channels * num_par_w * num_par_h;
    transform[NDIM-1][3] = out_channels * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part = runtime->get_logical_partition(
        ctx, weight->region, ip);
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
  if (weight->create_gradients && config.computationMode == COMP_MODE_TRAINING) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = weight->dims[i].size-1;
    hi[NDIM-1] = num_par_n * num_par_h * num_par_w * out_channels - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight->region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = out_channels-1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 4; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = out_channels;
    transform[NDIM-1][1] = out_channels * num_par_w;
    transform[NDIM-1][2] = out_channels * num_par_w * num_par_h;
    transform[NDIM-1][3] = out_channels * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight->part_grad = runtime->get_logical_partition(
        ctx, weight->region_grad, ip);
  }
}

template<int NDIM, int TDIM>
Tensor FFModel::create_linear_replica(const int dims[],
                                      const IndexSpaceT<TDIM>& task_is,
                                      DataType data_type)
{
  // No need to create replica for INFERENCE
  assert(config.computationMode == COMP_MODE_TRAINING);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  assert(NDIM >= 2);
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++)
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  Tensor replica = new TensorBase();
  replica->ts_guid = tensor_global_guid ++;
  replica->num_dims = NDIM;
  replica->data_type = data_type;
  for (int i = 0; i < NDIM; i++)
    replica->dims[i].size = dims[NDIM-1-i];
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator= runtime->create_field_allocator(ctx, fs);
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
  for (int i = 0; i < NDIM; i++)
    hi[i] = dims[NDIM-1-i]-1;
  Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  replica->region_grad = runtime->create_logical_region(ctx, is, fs);
  assert(dims[0] == num_parts[0]);
  //assert(dims[1] % num_parts[1] == 0);
  hi[NDIM-1] = dims[0] / num_parts[0] - 1; // replication dim
  hi[NDIM-2] = dims[1] / num_parts[TDIM-1] - 1; // sample dim
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
  Transform<NDIM, TDIM> transform;
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < TDIM; j++)
      transform[i][j] = 0;
  transform[NDIM-1][0] = hi[NDIM-1] + 1;
  transform[NDIM-2][TDIM-1] = hi[NDIM-2] + 1;
  //transform[NDIM-2][1] = dims[1] / num_parts[1];
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  replica->part_grad = runtime->get_logical_partition(
    ctx, replica->region_grad, ip);
  return replica;
}

IndexSpace FFModel::get_task_is(ParallelConfig pc) const
{
  std::map<ParallelConfig, IndexSpace, ParaConfigCompare>::const_iterator iter;
  iter = taskIs.find(pc);
  assert(iter != taskIs.end());
  return iter->second;
}

IndexSpace FFModel::get_or_create_task_is(const Tensor tensor)
{
  ParallelConfig pc;
  pc.nDims = 0;
  for (int i = 0; i < tensor->num_dims; i++)
    if (tensor->dims[i].parallel_idx >= 0) {
      pc.dim[tensor->dims[i].parallel_idx] = tensor->dims[i].degree;
      pc.nDims++;
    }
  if (pc.nDims == 0) {
    pc.nDims = 1;
    pc.dim[0] = 1;
  }
  return get_or_create_task_is(pc);
}

IndexSpace FFModel::get_or_create_task_is(ParallelConfig pc)
{
  if (taskIs.find(pc) != taskIs.end())
    return taskIs[pc];
  IndexSpace task_is;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  switch (pc.nDims) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> task_rect; \
      for (int i = 0; i < DIM; i++) { \
        task_rect.lo[i] = 0; \
        task_rect.hi[i] = pc.dim[i]-1; \
      } \
      task_is = runtime->create_index_space(ctx, task_rect); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  printf("ndim(%d) dims[%d %d %d %d]\n",
      pc.nDims, pc.dim[0], pc.dim[1], pc.dim[2], pc.dim[3]);
  taskIs[pc] = task_is;
  return task_is;
}

IndexSpace FFModel::get_or_create_task_is(const Domain& domain)
{
  ParallelConfig pc;
  pc.nDims = domain.get_dim();
  for (int i = 0; i < pc.nDims; i++) {
    pc.dim[i] = domain.hi()[i] - domain.lo()[i] + 1;
  }
  return get_or_create_task_is(pc);
}

IndexSpace FFModel::get_or_create_task_is(int ndims, const std::string& pcname)
{
  ParallelConfig pc;
  assert(config.find_parallel_config(ndims, pcname, pc));
  return get_or_create_task_is(pc);
}

IndexSpace FFModel::get_task_is(int ndims, const std::string& pcname) const
{
  ParallelConfig pc;
  assert(config.find_parallel_config(ndims, pcname, pc));
  return get_task_is(pc);
}

IndexSpace FFModel::get_task_is(const Domain& domain) const
{
  ParallelConfig pc;
  pc.nDims = domain.get_dim();
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = domain.hi()[i] - domain.lo()[i] + 1;
  std::map<ParallelConfig, IndexSpace, ParaConfigCompare>::const_iterator it;
  it = taskIs.find(pc);
  assert(it != taskIs.end());
  return it->second;
}

void FFModel::reset_metrics()
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID, TaskArgument(metrics_op, sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);
}

void FFModel::init_layers()
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->init(*this);
}

void FFModel::forward(int seq_length)
{
  iter_config.seq_length = seq_length;
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->forward(*this);
}

void FFModel::compute_metrics()
{
  Op* final_layer = layers[layers.size()-1];
  assert(final_layer->numOutputs == 1);
  metrics_op->compute(this, final_layer->outputs[0], label_tensor_with_final_part);
}

void FFModel::backward(int seq_length)
{
  iter_config.seq_length = seq_length;
  assert(config.computationMode == COMP_MODE_TRAINING);
  // Compute metrics
  Op* final_layer = layers[layers.size()-1];
  assert(final_layer->numOutputs == 1);
  metrics_op->compute(this, final_layer->outputs[0], label_tensor_with_final_part);
  // Compute the gradients of the final layer wrt loss
  loss_op->backward(this, final_layer->outputs[0], label_tensor_with_final_part);
  // Perform backpropagation
  // std::set<LogicalRegion> resetedInputGrads;
  for (int l = layers.size() - 1; l >= 0; l--) {
#ifdef ENABLE_RESNET_INPUT_GRADIENT_OPTIMIZATION
    for (int i = 0; i < layers[l]->numInputs; i++)
      if (resetedInputGrads.find(layers[l]->inputs[i]->region) == resetedInputGrads.end()) {
        resetedInputGrads.insert(layers[l]->inputs[i]->region);
      } else {
        // This input's gradients has been reseted by other layers
        // So we should not do it again
        layers[l]->resetInputGrads[i] = false;
      }
#endif
    layers[l]->backward(*this);
  }
}

void FFModel::update()
{
  optimizer->next();
  for (size_t i = 0; i < parameters.size(); i++) {
    optimizer->update(parameters[i]);
  }
}

void FFModel::compile(Optimizer* _optimizer,
                      LossType loss_type,
                      const std::vector<MetricsType>& metrics,
		      CompMode comp_mode)
{
  optimizer = _optimizer;
  compile(loss_type, metrics, comp_mode);
}

bool FFModel::apply_fusion(const std::vector<Op*>& layers,
                           std::vector<Op*>& new_layers)
{
  //Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  for (size_t l = 1; l < layers.size() - 1; l++) {
    size_t start = 0;
    {
      Op* opl = layers[l];
      for (int idx = 0; idx < opl->numInputs; idx++) {
        bool found = false;
        for (size_t i = 0; i < l; i++)
          if (opl->inputs[idx]->owner_op == layers[i]) {
            assert(!found);
            found = true;
            if (i > start) start = i;
          }
        assert(found || (opl->inputs[idx]->owner_op == NULL));
      }
    }
    for (size_t i = start; i < l; i++) {
      Domain d1 = runtime->get_index_space_domain(layers[l]->task_is);
      Domain d2 = runtime->get_index_space_domain(layers[i]->task_is);
      ParallelConfig pc1, pc2;
      assert(config.find_parallel_config(d1.get_dim(), layers[l]->name, pc1));
      assert(config.find_parallel_config(d2.get_dim(), layers[i]->name, pc2));
      if (pc1 == pc2) {
        FusedOp* fused_op;
        //bool created = false;
        if (layers[i]->op_type == OP_FUSED)
          fused_op = (FusedOp*) layers[i];
        else {
          //created = true;
          // cannot be an in-place operator
          if (layers[i]->has_inplace_output()) continue;
          fused_op = new FusedOp(*this, layers[i]);
        }
        if (fused_op->add_operator(*this, layers[l])) {
          // Construct new layers
          new_layers.clear();
          for (size_t j = 0; j < i; j++)
            new_layers.push_back(layers[j]);
          new_layers.push_back(fused_op);
          for (size_t j = i+1; j < layers.size(); j++) {
            if (j == l) continue; // l and i are fused
            Op* op = layers[j];
            // Update input tensors that belong to layer[l] or layer[i]
            for (int idx = 0; idx < op->numInputs; idx++) {
              if ((op->inputs[idx]->owner_op == layers[l])
              || (op->inputs[idx]->owner_op == layers[i]))
              {
                int found = -1;
                for (int k = 0; k < fused_op->numOutputs; k++)
                  if (fused_op->outputs[k]->region == op->inputs[idx]->region) {
                    assert(found == -1);
                    found = k;
                  }
                assert(found >= 0);
                op->inputs[idx] = fused_op->outputs[found];
              }
            }
            // Insert op
            new_layers.push_back(op);
          }
          // We are exact one layer fewer than the original
          assert(new_layers.size() + 1 == layers.size());
          return true;
        } else {
          //TODO: delete fused_op to avoid memory leakage
          //if (created)
            //delete fused_op;
          continue;
        }
      }
    }
  }
  return false;
}

void FFModel::compile(LossType loss_type,
                      const std::vector<MetricsType>& metrics,
                      CompMode comp_mode)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  config.computationMode = comp_mode;
  if (config.import_strategy_file.length() > 0) {
    load_strategies_from_file(config.import_strategy_file, config.strategies);
  }
  if (config.search_budget > 0) {
    // Launch the search task
    FFModel* model = this;
    TaskLauncher launcher(STRATEGY_SEARCH_TASK_ID,
        TaskArgument(&model, sizeof(FFModel*)));
    Future future = runtime->execute_task(ctx, launcher);
    future.get_void_result();
  } else {
    // Do nothing
  }

  loss_op = new Loss(loss_type);
  metrics_op = new Metrics(loss_type, metrics);

  // Init performance metrics
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID, TaskArgument(metrics_op, sizeof(Metrics)));
  current_metrics = runtime->execute_task(ctx, launcher);

  // Perform inplace optimizations
  for (size_t l = 1; l < layers.size(); l++) {
    if (layers[l]->can_inplace_output()) {
      // Assume outputs[0] is inplace with inputs[0]
      assert(layers[l]->numOutputs == 1);
      if (layers[l]->inputs[0]->owner_op != NULL) {
        int dim1 = layers[l]->outputs[0]->num_dims;
        int dim2 = layers[l]->inputs[0]->num_dims;
        ParallelConfig pc1, pc2;
        assert(config.find_parallel_config(dim1, layers[l]->name, pc1));
        assert(config.find_parallel_config(dim2, layers[l]->inputs[0]->owner_op->name, pc2));
        if (pc1 == pc2) {
          // Check no others also need layers[l]->inputs[0]
          bool found = false;
          for (size_t i = 0; i < layers.size(); i++) {
            if (i == l) continue;
            for (int j = 0; j < layers[i]->numInputs; j++) {
              if ((layers[i]->inputs[j]->owner_op == layers[l]->inputs[0]->owner_op)
              &&(layers[i]->inputs[j]->owner_idx == layers[l]->inputs[0]->owner_idx)) {
                found = true;
              }
            }
          }
          if (!found) {
            // Perform inplace
            layers[l]->do_inplace_output();
          }
        }
      }
    }
  }

  for (size_t l = 0; l < layers.size(); l++) {
    Op* op = layers[l];
    for (int i = 0; i < op->numInputs; i++) {
      if (op->inputs[i]->owner_op == NULL) {
        // Input tensor
        //assert(op->inputs[i]->sync_type == ParameterSyncType::NONE);
        map_tensor(op->inputs[i], op);
      } else {
        // No need to do anything else otherwise
      }
    }
    for (int i = 0; i < op->numWeights; i++) {
      if (op->weights[i]->owner_op == NULL) {
        // Weight tensor
        assert(op->weights[i]->owner_op == NULL);
        map_tensor(op->weights[i], op);
        parameters.push_back(op->weights[i]);
      }
    }
    for (int i = 0; i < op->numOutputs; i++) {
      // Output tensor
      map_tensor(op->outputs[i], op);
    }
    op->create_input_partition(*this);
    // op->map_output_tensors(*this);
  }

  // Check correctness
  for (size_t l = 0; l < layers.size(); l++) {
    Op* op = layers[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i]->owner_op == op);
      assert(op->outputs[i]->owner_idx == i);
      assert(op->outputs[i]->ts_guid != 0);
    }
  }

  // Perform fusion optimizations
  if (config.perform_fusion) {
    fprintf(stderr, "Applying fusion optimizations during compilation...\n");
    fprintf(stderr, "%zu layers before fusion...\n", layers.size());
    std::vector<Op*> new_layers;
    std::vector<Op*> old_layers = layers;
    while (apply_fusion(layers, new_layers)) {
      for (size_t i = 0; i < new_layers.size(); i++)
        for (int idx = 0; idx < new_layers[i]->numInputs; idx++)
          for (size_t j = i+1; j < new_layers.size(); j++)
            if (new_layers[i]->inputs[idx]->owner_op == new_layers[j])
              assert(false);
      layers = new_layers;
    }
    // Check integrity
    for (size_t l = 0; l < layers.size(); l++) {
      if (layers[l]->op_type == OP_FUSED) {
        FusedOp* fused = (FusedOp*) layers[l];
        int ioff = 0, woff = 0, ooff = 0;
        for (int op = 0; op < fused->numOperators; op++) {
          Op* old_op = fused->operators[op];
          for (int i = 0; i < fused->op_num_inputs[op]; i++) {
            int my_off = fused->op_input_idx[i+ioff];
            if (fused->op_input_source[i+ioff] == FusedOp::SOURCE_INPUT) {
              assert(fused->inputs[my_off]->region == old_op->inputs[i]->region);
            } else if (fused->op_input_source[i+ioff] == FusedOp::SOURCE_OUTPUT) {
              assert(fused->outputs[my_off]->region == old_op->inputs[i]->region);
            } else
              assert(false);
          }
          for (int i = 0; i < fused->op_num_weights[op]; i++) {
            int my_off = fused->op_weight_idx[i+woff];
            assert(fused->op_weight_source[i+woff] == FusedOp::SOURCE_WEIGHT);
            assert(fused->weights[my_off]->region == old_op->weights[i]->region);
          }
          for (int i = 0; i < fused->op_num_outputs[op]; i++) {
            int my_off = fused->op_output_idx[i+ooff];
            assert(fused->op_output_source[i+ooff] == FusedOp::SOURCE_OUTPUT);
            assert(fused->outputs[my_off]->region == old_op->outputs[i]->region);
          }
          ioff += fused->op_num_inputs[op];
          woff += fused->op_num_weights[op];
          ooff += fused->op_num_outputs[op];
        }
      } else {
        bool found = false;
        for (size_t i = 0; i < old_layers.size(); i++) {
          if (old_layers[i] == layers[l]) {
            assert(!found);
            found = true;
          }
        }
        assert(found);
      }
    }
    fprintf(stderr, "%zu layers after fusion...\n", layers.size());
    for (size_t i = 0; i < layers.size(); i++) {
        Op* op = layers[i];
        printf("layer[%zu]: type(%d)\n", i, layers[i]->op_type);
        for (int j = 0; j < op->numInputs; j++) {
          LogicalRegion handle = op->inputs[j]->region;
          printf("inputs[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                            handle.get_field_space().get_id(),
                            handle.get_tree_id());
        }
        for (int j = 0; j < op->numOutputs; j++) {
          LogicalRegion handle = op->outputs[j]->region;
          printf("outputs[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                            handle.get_field_space().get_id(),
                            handle.get_tree_id());
        }
        for (int j = 0; j < op->numWeights; j++) {
          LogicalRegion handle = op->weights[j]->region;
          printf("weights[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                            handle.get_field_space().get_id(),
                            handle.get_tree_id());
        }
    }
  }
  Op* final_layer = layers[layers.size()-1];
  // FIXME: currently assume the final layer has exactly one output
  assert(final_layer->numOutputs == 1);
  for (size_t i = 0; i < layers.size(); i++) {
      Op* op = layers[i];
      printf("layer[%zu]: type(%d)\n", i, layers[i]->op_type);
      for (int j = 0; j < op->numInputs; j++) {
        LogicalRegion handle = op->inputs[j]->region;
        printf("inputs[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                          handle.get_field_space().get_id(),
                          handle.get_tree_id());
      }
      for (int j = 0; j < op->numOutputs; j++) {
        LogicalRegion handle = op->outputs[j]->region;
        printf("outputs[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                          handle.get_field_space().get_id(),
                          handle.get_tree_id());
      }
  }
  //assert(final_layer->outputs[0].num_dims == 2);
  int dims[MAX_TENSOR_DIM], num_dims;
  num_dims = final_layer->outputs[0]->num_dims;
  // Note that FlexFlow's runtim internally reverse the array ordering
  for (int i = 0; i < num_dims; i++)
    dims[i] = final_layer->outputs[0]->dims[num_dims-1-i].size;
  DataType label_type = DT_FLOAT;
  if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    // assign dims[num_dims-1] = 1 for sparse categorical labels
    dims[num_dims-1] = 1;
    label_type = DT_INT32;
  }
  // create label tensor
  switch (num_dims) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      label_tensor = create_tensor<DIM>(dims, label_type); \
      map_tensor(label_tensor, label_tensor->owner_op); \
      label_tensor_with_final_part = label_tensor; \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      assert(false && "Unsupported dim");
    }
  }
  // init optimizer
  assert(optimizer != NULL);
  optimizer->init();
#ifdef FF_USE_NCCL
  if (config.computationMode == COMP_MODE_TRAINING) {
    // init all nccl communicators
    std::map<MappingTagID, ParallelConfig>::iterator iter;
    for (iter = config.strategies.begin(); iter != config.strategies.end(); iter++) {
      // only init nccl for GPU parallel configurations
      if (iter->second.device_type != ParallelConfig::GPU) continue;
      std::map<MappingTagID, ParallelConfig>::const_iterator it2;
      bool found = false;
      // Reuse nccl comms for same parallel config
      for (it2 = config.strategies.begin(); it2 != iter; it2++) {
        if (it2->second == iter->second) {
          found = true;
          for (int i = 0; i < it2->second.num_parts(); i++)
            iter->second.nccl_comms[i] = it2->second.nccl_comms[i];
        }
      }
      // Create new nccl comms
      if (!found) {
        TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
        Future future = runtime->execute_task(ctx, launcher);
        ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
        IndexSpace task_is = get_or_create_task_is(iter->second);
        ArgumentMap argmap;
        IndexLauncher index_launcher(NCCL_INIT_COMMS_TASK_ID, task_is,
            TaskArgument(&ncclId, sizeof(ncclUniqueId)), argmap,
            Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
            iter->first/*MappingTagID*/);
        FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
        fm.wait_all_results();
        int idx = 0;
        Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
        for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
          iter->second.nccl_comms[idx] = fm.get_result<ncclComm_t>(*it);
        }
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
void FFModel::propagate(std::map<Op*, ParallelConfig> const &current,
                        std::map<Op*, ParallelConfig> &next) const {
  next = current;
  size_t opId = std::rand() % (layers.size() - 1);
  //TODO: need to make sure opId is not an output layer of the model
  assert (opId != layers.size() - 1);

  std::vector<PropagationEdgeInfo> choosable_edges;
  std::unordered_set<Op *> opsSeen;

  auto bwd_edge_map = this->get_bwd_edge_map();

  Op *selected_op = this->layers[opId];
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
      edge_weights.push_back(
          FFModel::PROPAGATION_SIZE_WEIGHT * edge.size
            + avg_edge_size * (1 - FFModel::PROPAGATION_SIZE_WEIGHT)
      );
    }
    assert (edge_weights.size() == choosable_edges.size());
    PropagationEdgeInfo chosenEdgeInfo = select_random(choosable_edges, edge_weights);

    next[chosenEdgeInfo.dstOp] = next.at(selected_op);
    selected_op = chosenEdgeInfo.dstOp;
  } while (randf() < FFModel::CONTINUE_PROPAGATION_CHANCE);
}
#endif

void FFModel::rewrite(const std::map<const Op*, ParallelConfig>& current,
                      std::map<const Op*, ParallelConfig>& next,
                      bool use_propagation) const
{
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
    size_t opId = std::rand() % layers.size();
    //TODO: need to make sure opId is not an output layer of the model
    if (opId == layers.size() - 1)
      return;
    next[layers[opId]] = layers[opId]->get_random_parallel_config(*this);
  }
}

void FFModel::mcmc_optimize(std::map<const Op*, ParallelConfig>& best,
                            size_t budget, float alpha,
                            CompMode comp_mode,
                            bool use_propagation) const
{
  // Start from data parallel
  std::map<const Op*, ParallelConfig> current, next;
  float best_runtime = simulator->simulate_runtime(this, best, comp_mode);
  current = best;
  float current_runtime = best_runtime;
  size_t reset_span = budget / 100, last_reset_iter = 0;
  if (reset_span == 0)
    reset_span = 1;
  if (reset_span > 1000)
    reset_span = 1000;
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
      printf("iteration(%zu) current_strategy(%.4lf) best_strategy(%.4lf)\n", iter,
             current_runtime, best_runtime);
    }
    float rn = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    //float ratio = (next_runtime - current_runtime) / current_runtime;
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
  simulator->simulate_runtime(this, best, comp_mode, this->config.export_strategy_task_graph_file);
  std::map<const Op*, ParallelConfig>::const_iterator it;
  for (it = best.begin(); it != best.end(); it++) {
    printf("[%s] num_dims(%d) dims[", it->first->name, it->second.nDims);
    for (int i = 0; i < it->second.nDims; i++)
      if (i < it->second.nDims - 1)
        printf("%d,", it->second.dim[i]);
      else
        printf("%d", it->second.dim[i]);
    printf("] device_ids[");
    for (int i = 0; i < it->second.num_parts(); i++)
      if (i < it->second.num_parts() - 1)
        printf("%d,", it->second.device_ids[i]);
      else
        printf("%d", it->second.device_ids[i]);
    printf("]\n");
  }
  printf("============= MCMC Search Finished ============\n\n");
}

void FFModel::zero_gradients(void)
{
  for (int l = layers.size() - 1; l >= 0; l--)
    layers[l]->zero_grad(*this);
}

void FFModel::print_layers(int id)
{
  if (id == -1) {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->print_layer(*this);
    }
  } else {
    layers[id]->print_layer(*this);
  }
}

std::string FFModel::get_operator_type_name(OperatorType type) const
{
  switch(type) {
    case OP_CONV2D: return "Conv2D";
    case OP_DROPOUT: return "Dropout";
    case OP_LINEAR: return "Dense";
    case OP_BATCHMATMUL: return "BatchMatMul";
    case OP_POOL2D: return "Pool2D";
    case OP_RELU: return "ReLU";
    case OP_SIGMOID: return "Sigmoid";
    case OP_TANH: return "Tanh";
    case OP_ELU: return "Elu";
    case OP_FLAT: return "Flat";
    case OP_SOFTMAX: return "Softmax";
    case OP_BATCHNORM: return "BatchNorm";
    case OP_CONCAT: return "Concat";
    case OP_SPLIT: return "Split";
    case OP_EMBEDDING: return "Embedding";
    case OP_GROUP_BY: return "Group_by";
    case OP_AGGREGATE: return "Aggregate";
    case OP_RESHAPE: return "Reshape";
    case OP_REVERSE: return "Reverse";
    case OP_TRANSPOSE: return "Transpose";
    case OP_EW_ADD: return "Add";
    case OP_EW_MUL: return "Mul";
    case OP_MATMUL: return "Matmul";
    case OP_MUL: return "Mul";
    case OP_ENLARGE: return "Enlarge";
    case OP_SQUEEZE: return "Squeeze";
    case OP_UNSQUEEZE: return "Unsqueeze";
    case OP_EW_SUB: return "Sub";
    case OP_EW_DIV: return "Div";
    case OP_EW_EQUAL: return "Equal";
    case OP_EW_GREATER: return "Greater";
    case OP_EW_LESS: return "Less";
    case OP_EW_MAX: return "Max";
    case OP_EW_MIN: return "Min";
    case OP_REDUCE_ARGMAX: return "ReduceArgMax";
    case OP_REDUCE_ARGMIN: return "ReduceArgMin";
    case OP_REDUCE_MAX: return "ReduceMax";
    case OP_REDUCE_MEAN: return "ReduceMean";
    case OP_REDUCE_MIN: return "ReduceMin";
    case OP_REDUCE_PROD: return "ReduceProd";
    case OP_REDUCE_SUM: return "ReduceSum";
    case OP_PAD: return "Pad";
    case OP_SHAPE: return "Shape";
    case OP_SIZE: return "Size";
    case OP_TOPK: return "TopK";
    case OP_WHERE: return "Where";
    case OP_CEIL: return "Ceil";
    case OP_CAST: return "Cast";
    case OP_EXP: return "Exp";
    case OP_ROUND: return "Round";
    case OP_LOG: return "Log";
    case OP_LOGICAL_NOT: return "LogicalNot";
    case OP_SQRT: return "Sqrt";
    case OP_LEAKYRELU: return "LeakyReLU";
    case OP_SLICE: return "Slice";
    case OP_RESIZE: return "Resize";
    case OP_PRELU: return "PReLU";
    case OP_MULTIHEAD_ATTENTION: return "MultiHeadAttention";
    case OP_FUSED: return "FusedOp";
    // Parallel Ops
    case OP_REPARTITION: return "Repartition";
    case OP_COMBINE: return "Combine";
    case OP_REPLICATE: return "Replicate";
    case OP_REDUCTION: return "Reduction";
    case OP_PIPELINE: return "Pipeline";
    default: assert(false && "Not supported Operator type"); return "Unsupported";
  }
}

std::unordered_map<Op *, std::vector<std::pair<Op *, int>>> FFModel::get_bwd_edge_map() const {
  std::unordered_map<Op *, std::vector<std::pair<Op *, int>>> bwd_edge_map;
  for (auto const &layer : this->layers) {
    for (int i = 0; i < layer->numInputs; i++) {
      Op *src = (Op*) layer->inputs[i]->owner_op;
      bwd_edge_map[src].push_back({layer, layer->inputs[i]->get_volume()});
    }
  }

  return bwd_edge_map;
};

PerfMetrics FFModel::update_metrics_task(const Task *task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx, Runtime* runtime)
{
  Metrics* m = (Metrics*) task->args;
  //printf("in update_metrics_task\n");
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
  //fprintf(stderr, "acc_train_loss: %.4lf train_accuracy: %.2lf%%(%d/%d)\n",
  //        all_metrics.train_loss / all_metrics.train_all,
  //        all_metrics.train_correct * 100.0f / all_metrics.train_all,
  //        all_metrics.train_correct, all_metrics.train_all);
  return all_metrics;
}

void Op::prefetch(const FFModel& ff)
{
  // TODO: perform prefetch for performance imporvement
}

#ifdef DEADCODE
// ========================================================
// class DataLoader
// ========================================================
DataLoader::DataLoader(std::string datasetPath)
{
  std::string trainPath = datasetPath + "/train";
  std::string valPath = datasetPath + "/val";
  DIR* trainDir = opendir(trainPath.c_str());
  DIR* valDir = opendir(valPath.c_str());
  if (!trainDir) {
    log_model.print("Failed to open %s\n", trainPath.c_str());
    return;
  }
  if (!valDir) {
    log_model.print("Failed to open %s\n", valPath.c_str());
    return;
  }
  for (struct dirent* dp = readdir(trainDir); dp; dp = readdir(trainDir)) {
    std::string labelId(dp->d_name);
    if (labelId == "." || labelId == "..")
      continue;
    DIR* labelDir = opendir((trainPath + "/" + labelId).c_str());
    if (!labelDir)
      continue;
    for (struct dirent* sp = readdir(labelDir); sp; sp = readdir(labelDir)) {
      std::string sampleId(sp->d_name);
      if (sampleId == "." || sampleId == "..")
        continue;

    }
    printf("%s/%s\n", trainPath.c_str(), labelId.c_str());
    closedir(labelDir);
  }
  closedir(trainDir);
  closedir(valDir);
}

bool DataLoader::get_samples(int numSamples, DataLoadMeta &meta)
{
  meta.numSamples = numSamples;
  for (int i = 0; i < numSamples; i++) {
    if (sampleIter == samples.end())
      sampleIter = samples.begin();
    meta.samples[i] = *sampleIter;
  }
  return true;
}

bool DataLoader::shuffle_samples(void)
{
  std::random_shuffle(samples.begin(), samples.end());
  return true;
}
#endif

// ========================================================
// class FFIterationConfig
// ========================================================
FFIterationConfig::FFIterationConfig()
{
  seq_length = -1;
}

void FFIterationConfig::reset()
{
  seq_length = -1;
}

// ========================================================
// class FFConfig
// ========================================================

// Default Config Parameters
struct DefaultConfig {
  const static int epochs = 1;
  //const static int iterations = 1;
  const static int batchSize = 64;
  const static bool profiling = false;
  constexpr static float learningRate = 0.01f;
  constexpr static float weightDecay = 0.0001f;
  const static size_t workSpaceSize = (size_t)1 * 1024 * 1024 * 1024; // 2GB
  const static int numNodes = 1;
  const static int workersPerNode = 0;
  const static int cpusPerNode = 0;
  const static size_t searchBudget = 0;
  const static size_t simulatorWorkSpaceSize = (size_t)2 * 1024 * 1024 * 1024; //2GB
  constexpr static float searchAlpha = 1.0f;
  const static bool searchOverlapBackwardUpdate = false;
  const static bool enableSampleParallel = true;
  const static bool enableParameterParallel = false;
  const static bool enableAttributeParallel = false;
  const static bool allowTensorOpMathConversion = false;
  const static int machine_model_version = 0;
  const static int simulator_segment_size = 16777216; // 16 MB
  const static int simulator_max_num_segments = 1;
};

FFConfig::FFConfig()
{
  epochs = DefaultConfig::epochs;
  //iterations = DefaultConfig::iterations;
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
  enable_sample_parallel = DefaultConfig::enableSampleParallel;
  enable_parameter_parallel = DefaultConfig::enableParameterParallel;
  enable_attribute_parallel = DefaultConfig::enableAttributeParallel;
  allow_tensor_op_math_conversion = DefaultConfig::allowTensorOpMathConversion;
  machine_model_version = DefaultConfig::machine_model_version;
  simulator_segment_size = DefaultConfig::simulator_segment_size;
  simulator_max_num_segments = DefaultConfig::simulator_max_num_segments;
  machine_model_file = "";
  import_strategy_file = "";
  export_strategy_file = "";
  export_strategy_task_graph_file = "";
  export_strategy_computation_graph_file = "";
  dataset_path = "";
  syntheticInput = false;
  perform_fusion = false;

  // Parse input arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_args(argv, argc);
  }

  Runtime *runtime = Runtime::get_runtime();
  lg_hlr = runtime;
  lg_ctx = Runtime::get_context();
  field_space = runtime->create_field_space(lg_ctx);
}

void FFConfig::parse_args(char **argv, int argc)
{
  for (int i = 1; i < argc; i++)
  {
    if ((!strcmp(argv[i], "-e")) || (!strcmp(argv[i], "--epochs"))) {
      epochs = atoi(argv[++i]);
      continue;
    }
    //if ((!strcmp(argv[i], "-i")) || (!strcmp(argv[i], "--iterations"))) {
    //  iterations = atoi(argv[++i]);
    //  continue;
    //}
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
    if ((!strcmp(argv[i], "--budget")) || (!strcmp(argv[i], "--search-budget"))) {
      search_budget =(size_t) atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--alpha")) || (!strcmp(argv[i], "--search-alpha"))) {
      search_alpha = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--simulator-workspace-size"))
    {
      simulator_work_space_size = atoll(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--import")) || (!strcmp(argv[i], "--import-strategy"))) {
      import_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--export")) || (!strcmp(argv[i], "--export-strategy"))) {
      export_strategy_file = std::string(argv[++i]);
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
    if (!strcmp(argv[i], "-ll:gpu"))
    {
      workersPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--nodes"))
    {
      numNodes = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ll:cpu"))
    {
      cpusPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--profiling"))
    {
      profiling = true;
      continue;
    }
    if (!strcmp(argv[i], "--allow-tensor-op-math-conversion"))
    {
      allow_tensor_op_math_conversion = true;
      continue;
    }
    if (!strcmp(argv[i], "--fusion"))
    {
      perform_fusion = true;
      continue;
    }
    if (!strcmp(argv[i], "--overlap"))
    {
      search_overlap_backward_update = true;
      continue;
    }
    if (!strcmp(argv[i], "--taskgraph")) {
      export_strategy_task_graph_file = std::string(argv[++i]);
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
  }
}

void register_flexflow_internal_tasks()
{
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
    TaskVariantRegistrar registrar(ELEMENTUNARY_INIT_TASK_ID, "ElementWiseUnary Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, ElementUnary::init_task>(
        registrar, "ElementWiseUnary Init Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_FWD_TASK_ID, "ElementWiseUnary Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementUnary::forward_task>(
        registrar, "ElementWiseUnary Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTUNARY_BWD_TASK_ID, "ElementWiseUnary Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementUnary::backward_task>(
        registrar, "ElementWiseUnary Backward Task");
  }
  // ElementBinary task
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_INIT_TASK_ID, "ElementWiseBinary Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, ElementBinary::init_task>(
        registrar, "ElementWiseBinary Init Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_FWD_TASK_ID, "ElementWiseBinary Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementBinary::forward_task>(
        registrar, "ElementWiseBinary Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ELEMENTBINARY_BWD_TASK_ID, "ElementWiseBinary Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ElementBinary::backward_task>(
        registrar, "ElementWiseBinary Backward Task");
  }
  // Conv2D task
  {
    TaskVariantRegistrar registrar(CONV2D_INIT_TASK_ID, "Conv2D Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Conv2D::init_task>(
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
    Runtime::preregister_task_variant<OpMeta*, Dropout::init_task>(
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
    Runtime::preregister_task_variant<OpMeta*, Embedding::init_task>(
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

  // Group by task CPU
  {
    TaskVariantRegistrar registrar(GROUP_BY_INIT_TASK_ID, "Group_by Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Group_by::init_task>(
        registrar, "Group_by Init Task");
  }
  {
    TaskVariantRegistrar registrar(GROUP_BY_FWD_TASK_ID, "Group_by Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Group_by::forward_task>(
        registrar, "Group_by Forward Task");
  }
  {
    TaskVariantRegistrar registrar(GROUP_BY_BWD_TASK_ID, "Group_by Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Group_by::backward_task>(
        registrar, "Group_by Backward Task");
  }

  // Aggregate task CPU
  {
    TaskVariantRegistrar registrar(AGGREGATE_INIT_TASK_ID, "Aggregate Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Aggregate::init_task>(
        registrar, "Aggregate Init Task");
  }
  {
    TaskVariantRegistrar registrar(AGGREGATE_FWD_TASK_ID, "Aggregate Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Aggregate::forward_task>(
        registrar, "Aggregate Forward Task");
  }
  {
    TaskVariantRegistrar registrar(AGGREGATE_BWD_TASK_ID, "Aggregate Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Aggregate::backward_task>(
        registrar, "Aggregate Backward Task");
  }

  // Pool2D task
  {
    TaskVariantRegistrar registrar(POOL2D_INIT_TASK_ID, "pool2d_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Pool2D::init_task>(
        registrar, "pool2d_init_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_FWD_TASK_ID, "pool2d_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pool2D::forward_task>(
        registrar, "pool2d_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(POOL2D_BWD_TASK_ID, "pool2d_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Pool2D::backward_task>(
        registrar, "pool2d_bwd_task");
  }
  // BatchNorm task
  {
    TaskVariantRegistrar registrar(BATCHNORM_INIT_TASK_ID, "bn_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, BatchNorm::init_task>(
        registrar, "bn_init_task");
  }
  {
    TaskVariantRegistrar registrar(BATCHNORM_FWD_TASK_ID, "bn_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::forward_task>(
        registrar, "bn_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(BATCHNORM_BWD_TASK_ID, "bn_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::backward_task>(
        registrar, "bn_bwd_task");
  }
  // BatchMatmul task
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_INIT_TASK_ID, "BatchMatmul Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, BatchMatmul::init_task>(
        registrar, "BatchMatmul Init Task");
  }
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_FWD_TASK_ID, "BatchMatmul Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchMatmul::forward_task>(
        registrar, "BatchMatmul Forward Task");
  }
  {
    TaskVariantRegistrar registrar(BATCHMATMUL_BWD_TASK_ID, "BatchMatmul Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchMatmul::backward_task>(
        registrar, "BatchMatmul Backward Task");
  }
  // Linear task
  {
    TaskVariantRegistrar registrar(LINEAR_INIT_TASK_ID, "Linear Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Linear::init_task>(
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
  {
    TaskVariantRegistrar registrar(LINEAR_BWD2_TASK_ID,
                                   "Linear Backward (Aggregate replica)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Linear::backward2_task>(
        registrar, "Linear Backward Task (Aggregate replica)");
  }
  // Flat task
  {
    TaskVariantRegistrar registrar(FLAT_INIT_TASK_ID, "flat_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Flat::init_task>(
        registrar, "flat_init_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_FWD_TASK_ID, "flat_fwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::forward_task>(
        registrar, "flat_fwd_task");
  }
  {
    TaskVariantRegistrar registrar(FLAT_BWD_TASK_ID, "flat_bwd_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Flat::backward_task>(
        registrar, "flat_bwd_task");
  }
  // Softmax task
  {
    TaskVariantRegistrar registrar(SOFTMAX_INIT_TASK_ID, "softmax_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Softmax::init_task>(
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
    TaskVariantRegistrar registrar(METRICS_COMP_TASK_ID, "MSELoss Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<PerfMetrics, Metrics::compute_task>(
        registrar, "MSELoss Backward Task");
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
    Runtime::preregister_task_variant<PerfMetrics, FFModel::update_metrics_task>(
        registrar, "Update Metrics Task");
  }
  // Concat task
  {
    TaskVariantRegistrar registrar(CONCAT_INIT_TASK_ID, "Concat Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Concat::init_task>(
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
    Runtime::preregister_task_variant<OpMeta*, Split::init_task>(
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
  // Reshape task
  {
    TaskVariantRegistrar registrar(RESHAPE_INIT_TASK_ID, "Reshape Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Reshape::init_task>(
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
    Runtime::preregister_task_variant<OpMeta*, Reverse::init_task>(
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
  // Reverse task
  {
    TaskVariantRegistrar registrar(TOPK_INIT_TASK_ID, "TopK Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, TopK::init_task>(
        registrar, "TopK Init Task");
  }
  {
    TaskVariantRegistrar registrar(TOPK_FWD_TASK_ID, "TopK Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TopK::forward_task>(
        registrar, "TopK Forward Task");
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
    Runtime::preregister_task_variant<OpMeta*, Transpose::init_task>(
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
    TaskVariantRegistrar registrar(ATTENTION_INIT_TASK_ID, "MultiHeadAttention Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, MultiHeadAttention::init_task>(
        registrar, "MultiHeadAttention Init Task");
  }
  {
    TaskVariantRegistrar registrar(ATTENTION_FWD_TASK_ID, "MultiHeadAttention Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<MultiHeadAttention::forward_task>(
        registrar, "MultiHeadAttention Forward Task");
  }
  {
    TaskVariantRegistrar registrar(ATTENTION_BWD_TASK_ID, "MultiHeadAttention Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<MultiHeadAttention::backward_task>(
        registrar, "MultiHeadAttention Backward Task");
  }
  // FusedOp Task
  {
    TaskVariantRegistrar registrar(FUSEDOP_INIT_TASK_ID, "FusedOp Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, FusedOp::init_task>(
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
    TaskVariantRegistrar registrar(REPARTITION_FWD_TASK_ID, "Repartition Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Repartition::forward_task>(
        registrar, "Repartition Forward Task");
  }
  {
    TaskVariantRegistrar registrar(REPARTITION_BWD_TASK_ID, "Repartition Backward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Repartition::backward_task>(
        registrar, "Repartition Backward Task");
  }
  // Combine
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
    TaskVariantRegistrar registrar(FUSED_PARALLELOP_FWD_TASK_ID, "FusedParallel Forward");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<FusedParallelOp::forward_task>(
        registrar, "FusedParallel Forward Task");
  }
  {
    TaskVariantRegistrar registrar(FUSED_PARALLELOP_BWD_TASK_ID, "FusedParallel Backward");
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
    TaskVariantRegistrar registrar(SGD_UPD_NCCL_TASK_ID,
                                   "SGD NCCL Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SGDOptimizer::nccl_update_task>(
        registrar, "SGD NCCL Update Task");
  }
  {
    TaskVariantRegistrar registrar(ADAM_UPD_NCCL_TASK_ID,
                                   "Adam NCCL Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AdamOptimizer::nccl_update_task>(
        registrar, "Adam NCCL Update Task");
  }
#endif
  // Initializer
  {
    TaskVariantRegistrar registrar(ZERO_INIT_TASK_ID,
                                   "Zero Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ZeroInitializer::init_task_cpu>(
        registrar, "Zero Init Task");
  }
  {
    TaskVariantRegistrar registrar(ZERO_INIT_TASK_ID,
                                   "Zero Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ZeroInitializer::init_task>(
        registrar, "Zero Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONSTANT_INIT_TASK_ID,
                                   "Constant Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ConstantInitializer::init_task_cpu>(
        registrar, "Constant Init Task");
  }
  {
    TaskVariantRegistrar registrar(CONSTANT_INIT_TASK_ID,
                                   "Constant Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ConstantInitializer::init_task>(
        registrar, "Constant Init Task");
  }
  {
    TaskVariantRegistrar registrar(UNIFORM_INIT_TASK_ID,
                                   "Uniform Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UniformInitializer::init_task>(
        registrar, "Uniform Init Task");
  }
  {
    TaskVariantRegistrar registrar(GLOROT_INIT_TASK_ID,
                                   "Glorot Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<GlorotUniform::init_task>(
        registrar, "Glorot Init Task");
  }
  {
    TaskVariantRegistrar registrar(NORMAL_INIT_TASK_ID,
                                   "Normalize Init");
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
    Runtime::preregister_task_variant<ncclUniqueId, Op::get_nccl_unique_id_task>(
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
    TaskVariantRegistrar registrar(STRATEGY_SEARCH_TASK_ID,
                                   "Stretegy Search");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Simulator::strategy_search_task>(
        registrar, "Stretegy Search Task");
  }
  // Parameter Server Prefetch task
  {
    TaskVariantRegistrar registrar(PS_PREFETCH_TASK_ID, "Weights Prefetch");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UtilityTasks::dummy_task>(registrar, "Weights Prefetch Task");
  }
}

// template instantiations
#define DIMFUNC(DIM) \
  template Tensor FFModel::create_tensor<DIM>(const int dims[], DataType data_type, const Op* owner_op, int owner_idx, bool create_grad); \
  template Tensor FFModel::create_tensor<DIM>(const ParallelDim dims[], DataType data_type, const Op* owner_op, int owner_idx, bool create_grad); \
  template Parameter FFModel::create_weight<DIM>(const int dims[], DataType data_type, const Op* owner_op, bool create_grad,\
    Initializer* initializer, ParameterSyncType sync_type);\
  template Parameter FFModel::create_weight<DIM>(const ParallelDim dims[], DataType data_type, const Op* owner_op, bool create_grad,\
    Initializer* initializer, ParameterSyncType sync_type);\
  template void FFModel::map_tensor_with_dim<DIM>(Tensor tensor, const Op* parallel_op); \
  template void FFModel::map_weight_with_dim<DIM>(Tensor weight, const Op* parallel_op); \
  template Tensor FFModel::create_constant<DIM>(const int* dims, float value, DataType data_type); \
  template void FFModel::create_disjoint_partition<DIM>(const Tensor tensor, const IndexSpaceT<DIM>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
  LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC

#define DIMFUNC(D1,D2) \
  template void FFModel::map_tensor_with_dim2<D1,D2>(Tensor tensor, const Op* parallel_op); \
  template void FFModel::create_disjoint_partition_with_dim2<D1,D2>(const ParallelDim dims[], const IndexSpaceT<D2>& part_is, const LogicalRegion& region, LogicalPartition& part); \
  template void FFModel::create_aliased_partition_with_dim2<D1,D2>(const ParallelDim dims[], int aliased_dim, const IndexSpaceT<D2>& part_is, const LogicalRegion& region, LogicalPartition& part); \
  template void FFModel::create_data_parallel_partition_with_diff_dims<D1, D2>(const Tensor tensor, const IndexSpaceT<D2>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
  LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

template void FFModel::map_conv_weight<4>(Tensor weight, const Op* parallel_op);
template void FFModel::map_conv_weight<1>(Tensor weight, const Op* parallel_op);

#define DIMFUNC(D1,D2) \
  template void FFModel::map_linear_weight<D1, D2>(Tensor p, const Op* op);
  LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

#define DIMFUNC(D1,D2) \
  template Tensor FFModel::create_linear_replica<D1>(const int* dims, const IndexSpaceT<D2>& part_is, DataType data_type);
  LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

template float* TensorBase::get_raw_ptr<float>(FFConfig &config);
template int32_t* TensorBase::get_raw_ptr<int32_t>(FFConfig &config);
