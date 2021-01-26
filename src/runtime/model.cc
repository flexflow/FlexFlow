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

using namespace std;

LegionRuntime::Logger::Category log_model("ff");

Tensor::Tensor(void)
{
  numDim = 0;
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    adim[i] = 0;
    //pdim[i] = 0;
  }
  region = LogicalRegion::NO_REGION;
  region_grad = LogicalRegion::NO_REGION;
  part = LogicalPartition::NO_PART;
  part_grad = LogicalPartition::NO_PART;
  owner_op = NULL;
  owner_idx = 0;
  //physical_region.impl = NULL;
}

Tensor& Tensor::operator=(const Tensor& rhs)
{
  numDim = rhs.numDim;
  for (int i = 0; i < numDim; i++)
    adim[i] = rhs.adim[i];
  data_type = rhs.data_type;
  owner_op = rhs.owner_op;
  owner_idx = rhs.owner_idx;
  region = rhs.region;
  region_grad = rhs.region_grad;
  part = rhs.part;
  part_grad = rhs.part_grad;
  physical_region = rhs.physical_region;
  return *this;
}

void Tensor::inline_map(FFConfig &config)
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

void Tensor::inline_unmap(FFConfig &config)
{
  printf("inline unmap tensor\n");
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  assert(physical_region.is_valid() == true);
  runtime->unmap_region(ctx, physical_region);
}

template<typename T>
T* Tensor::get_raw_ptr(FFConfig &config)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  T *raw_ptr = NULL;
  if (numDim == 1) {
    TensorAccessorW<T, 1> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (numDim == 2) {
    TensorAccessorW<T, 2> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (numDim == 3) {
    TensorAccessorW<T, 3> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (numDim == 4) {
    TensorAccessorW<T, 4> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else {
    printf("wrong numDim %d", numDim);
    assert(0);
  }
  return raw_ptr;
}

void Tensor::attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major)
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

void Tensor::detach_raw_ptr(FFConfig &config)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  runtime->detach_external_resource(ctx, physical_region);
}

bool Tensor::get_input_sub_tensor(const ParallelConfig& pc,
                                  Tensor& tensor,
                                  OperatorType type)
{
  //TODO: consider reduction dim for conv2d and linear
  switch (type) {
    case OP_FLAT:
      {
        assert (pc.nDims == 2 && "Invalid dimension for parallel config of OP_FLAT");
        int nonBatchDim = pc.dim[0];
        tensor.numDim = numDim;
        assert (nonBatchDim == 1 && "I'm not sure this is correct otherwise");
        if (adim[numDim - 1] % nonBatchDim != 0) {
          printf("Could not get input subtensor because the dimension is not divisiable: %d %% %d != 0\n", adim[numDim - 1], nonBatchDim);
        }
        for (int i = numDim - 2; i >= 0; i--) {
          tensor.adim[i] = adim[i];
        }
      }
      break;
    default:
      {
        if (pc.nDims != numDim) {
          printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, numDim);
          return false;
        }
        for (int i = 0; i < numDim; i++) {
          if (adim[i] % pc.dim[i] != 0) {
            printf("Could not get input subtensor because the given dimension is not divisible: %d %% %d != 0\n", adim[i], pc.dim[i]);
            return false;
          }
        }
        tensor.numDim = numDim;
        for (int i = 0; i < numDim; i++) {
          tensor.adim[i] = adim[i] / pc.dim[i];
        }
        tensor.data_type = data_type;
      }
      break;
  }
  return true;
}

bool Tensor::get_output_sub_tensor(const ParallelConfig& pc,
                                   Tensor& tensor,
                                   OperatorType type)
{
  if (pc.nDims != numDim) {
    printf("Could not get output subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, numDim);
    return false;
  }
  for (int i = 0; i < numDim; i++) {
    if (adim[i] % pc.dim[i] != 0) {
      printf("Could not get output subtensor because the given dimension is not divisible: %d %% %d != 0\n", adim[i], pc.dim[i]);
      return false;
    }
  }
  tensor.numDim = numDim;
  for (int i = 0; i < numDim; i++)
    tensor.adim[i] = adim[i] / pc.dim[i];
  tensor.data_type = data_type;
  return true;
}

size_t Tensor::get_volume() const
{
  size_t volume = 1;
  for (int i = 0; i < numDim; i++)
    volume *= adim[i];
  return volume;
}

Domain Tensor::get_domain() const
{
  Domain d;
  d.dim = this->numDim;
  for (int i = 0; i < this->numDim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+Domain::MAX_RECT_DIM] = this->adim[i] - 1;
  }
  return d;
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       const Tensor& _input)
: op_type(_op_type), numInputs(1), numWeights(0), numOutputs(1)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  inputs[0] = _input;
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
    outputs[i].data_type = inputs[0].data_type;
  }
#ifdef FF_ENABLE_NCCL
  get_nccl_unique_id();
#endif
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const Op* shared_op,
       const char* _name,
       const Tensor& _input)
: op_type(_op_type), numInputs(1), numWeights(0), numOutputs(1)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  if (shared_op == NULL) {
    pcname = pcname + "_" + std::to_string(model.op_global_guid++);
  } else {
    pcname = std::string(shared_op->name);
  }
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  inputs[0] = _input;
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
    outputs[i].data_type = inputs[0].data_type;
  }
#ifdef FF_ENABLE_NCCL
  get_nccl_unique_id();
#endif
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       const Tensor& _input1,
       const Tensor& _input2)
: op_type(_op_type), numInputs(2), numWeights(0), numOutputs(1)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  inputs[0] = _input1;
  inputs[1] = _input2;
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
    outputs[i].data_type = inputs[0].data_type;
  }
#ifdef FF_ENABLE_NCCL
  get_nccl_unique_id();
#endif
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       const Tensor& _input1,
       const Tensor& _input2,
       const Tensor& _input3)
: op_type(_op_type), numInputs(3), numWeights(0), numOutputs(1)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  inputs[0] = _input1;
  inputs[1] = _input2;
  inputs[2] = _input3;
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
    outputs[i].data_type = inputs[0].data_type;
  }
#ifdef FF_ENABLE_NCCL
  get_nccl_unique_id();
#endif
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       int n, const Tensor* _inputs)
: op_type(_op_type), numInputs(n), numWeights(0), numOutputs(1)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  assert(n <= MAX_NUM_INPUTS);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
    outputs[i].data_type = inputs[0].data_type;
  }
#ifdef FF_ENABLE_NCCL
  get_nccl_unique_id();
#endif
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const char* _name,
       int _numInputs)
: op_type(_op_type), numInputs(_numInputs), numWeights(0), numOutputs(1)
{
  std::string pcname;
  if (_name == NULL) {
    pcname = model.get_operator_type_name(op_type);
  } else {
    pcname = std::string(_name);
  }
  pcname = pcname + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
    outputs[i].data_type = inputs[0].data_type;
  }
#ifdef FF_ENABLE_NCCL
  get_nccl_unique_id();
#endif
}

Parameter* Op::get_parameter(int index)
{
  assert(index < numWeights);
  return &weights[index];
}

void Op::zero_grad(const FFModel& ff)
{
  Runtime* runtime = ff.config.lg_hlr;
  Context ctx = ff.config.lg_ctx;
  ArgumentMap argmap;
  IndexLauncher launcher(ZERO_INIT_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  for (int i = 0; i < numWeights; i++) {
    launcher.add_region_requirement(
        RegionRequirement(weights[i].part_grad, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, weights[i].region_grad));
    launcher.add_field(i, FID_DATA);
  }
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
        RegionRequirement(outputs[i].part_grad, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, outputs[i].region_grad));
    //LogicalRegion lr = outputs[i].region_grad;
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
  pc.nDims = outputs[0].numDim;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = i;
  return pc;
}

ParallelConfig Op::get_random_parallel_config(const FFModel& ff) const
{
  std::vector<int> candidates;
  int batch_size = outputs[0].adim[outputs[0].numDim-1];
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
  pc.nDims = outputs[0].numDim;
  for (int i = 0; i < pc.nDims; i++)
    pc.dim[i] = i == pc.nDims - 1 ? num_parts : 1;
  int total_num_devices = ff.config.workersPerNode * ff.config.numNodes;
  int start_idx = std::rand() % (total_num_devices - num_parts + 1);
  for (int i = 0; i < num_parts; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

Domain Op::get_output_tensor_shape(const ParallelConfig& pc,
                                   int output_idx, int part_idx)
{
  assert(output_idx < numOutputs);
  Domain d;
  d.dim = outputs[output_idx].numDim;
  // Assume pc dim matches output dim
  assert(d.dim == pc.nDims);
  for (int i = 0; i < d.dim; i++) {
    // Assume an equal partitioning
    assert(outputs[output_idx].adim[i] % pc.dim[i] == 0);
    int dim_size = outputs[output_idx].adim[i] / pc.dim[i];
    d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
    d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    part_idx = part_idx / pc.dim[i];
  }
  assert(part_idx == 0);
  return d;
}

Domain Op::get_input_tensor_shape(const ParallelConfig& pc,
                                  int input_idx, int part_idx)
{
  assert(input_idx < numInputs);
  Domain d;
  d.dim = inputs[input_idx].numDim;
  if (pc.nDims == d.dim) {
    for (int i = 0; i < d.dim; i++) {
      // Assume an equal partitioning
      assert(inputs[input_idx].adim[i] % pc.dim[i] == 0);
      int dim_size = inputs[input_idx].adim[i] / pc.dim[i];
      d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
      part_idx = part_idx / pc.dim[i];
    }
  } else {
    // Require data parallel when dims mismatch
    for (int i = 0; i < pc.nDims-1; i++)
      assert(pc.dim[i] == 1);
    for (int i = 0; i < d.dim-1; i++) {
      int dim_size = inputs[input_idx].adim[i];
      d.rect_data[i] = 0;
      d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    }
    // Assume an equal partitioning
    assert(inputs[input_idx].adim[d.dim-1] % pc.dim[pc.nDims-1] == 0);
    assert(part_idx < pc.dim[pc.nDims-1]);
    int dim_size = inputs[input_idx].adim[d.dim-1] / pc.dim[pc.nDims-1];
    d.rect_data[d.dim - 1] = part_idx * dim_size;
    d.rect_data[2*d.dim - 1] = d.rect_data[d.dim-1] + dim_size - 1;
    part_idx = part_idx / pc.dim[pc.nDims-1];
  }
  assert(part_idx == 0);
  return d;
}

Domain Op::get_weight_tensor_shape(const ParallelConfig& pc,
                                   int weight_idx, int part_idx)
{
  // Default data parallel weight replication
  assert(weight_idx < numWeights);
  Domain d;
  d.dim = weights[weight_idx].numDim;
  for (int i = 0; i < d.dim; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = weights[weight_idx].adim[i] - 1;
  }
  return d;
}

#ifdef FF_ENABLE_NCCL  
void Op::get_nccl_unique_id()
{
  // Init NCCL id
  int my_rank = -1, all_ranks = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &all_ranks);
  if (my_rank == 0) ncclGetUniqueId(&ncclId);
  MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  //fprintf(stderr, "In Op(%p): MPImyrank(%d) MPIallranks(%d) ncclID(%p)\n",
  //    this, my_rank, all_ranks, ncclId);
}
#endif

OpMeta::OpMeta(FFHandler _handle)
: handle(_handle)
{}

#ifdef FF_ENABLE_NCCL
void OpMeta::init_nccl_communicator(const Task* task,
                                    ncclUniqueId ncclId)
{
  // Must be an index space launch
  assert(task->is_index_space);
  int allRanks = task->index_domain.get_volume();
  assert(task->index_domain.contains(task->index_point));
  int myRank = 0;
  for (Domain::DomainPointIterator it(task->index_domain); it; it++, myRank++) {
    if (it.p == task->index_point) break;
  }
  checkNCCL(ncclCommInitRank(&ncclComm, allRanks, ncclId, myRank));
  //fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //    ncclComm, allRanks, myRank, ncclId);
}
#endif

FFModel::FFModel(FFConfig& _config)
: op_global_guid(100), config(_config),
  optimizer(NULL), loss_op(NULL), metrics_op(NULL)
{
  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
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
      RegionRequirement(tensor.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, tensor.region));
  launcher.add_field(0, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return tensor;
}

template<int NDIM>
Tensor FFModel::create_tensor(const int dims[],
                              DataType data_type,
                              const Op* owner_op,
                              bool create_grad)
{
  Tensor tensor;
  tensor.data_type = data_type;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;

  std::string name = "";
  if (owner_op != NULL)
    name = std::string(owner_op->name);
  IndexSpaceT<NDIM> part_is = (IndexSpaceT<NDIM>) get_or_create_task_is(NDIM, name);
  // Step 1: create regions
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator= runtime->create_field_allocator(ctx, fs);
  switch (data_type)
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
    hi[i] = dims[NDIM-1-i]-1;
  Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  tensor.region = runtime->create_logical_region(ctx, is, fs);
  if (create_grad) {
    tensor.region_grad = runtime->create_logical_region(ctx, is, fs);
  }

  // Step 2: create partitions
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
      ctx, is, part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  tensor.part = runtime->get_logical_partition(ctx, tensor.region, ip);
  if (create_grad) {
    tensor.part_grad = runtime->get_logical_partition(ctx, tensor.region_grad, ip);
  }
  tensor.numDim = NDIM;
  for (int i = 0; i < NDIM; i++) {
    tensor.adim[i] = rect.hi[i] - rect.lo[i] + 1;
    //tensor.pdim[i] = extent.hi[i] - extent.lo[i] + 1;
  }

#ifdef DEADCODE
  // Initialize tensor with zero
  ArgumentMap argmap;
  IndexLauncher launcher(ZERO_INIT_TASK_ID, part_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false, 0,
                         FFConfig::get_hash_id(name));
  launcher.add_region_requirement(
      RegionRequirement(tensor.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, tensor.region));
  launcher.add_field(0, FID_DATA);
  if (create_grad) {
    launcher.add_region_requirement(
        RegionRequirement(tensor.part_grad, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, tensor.region_grad));
    launcher.add_field(1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
#endif
  return tensor;
}

template<int NDIM>
void FFModel::create_disjoint_partition(const Tensor& tensor,
                                        const IndexSpaceT<NDIM>& part_is,
                                        LogicalPartition& part_fwd,
                                        LogicalPartition& part_bwd)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  // Check that dimension sizes match
  {
    assert(tensor.numDim == NDIM);
    Domain domain = runtime->get_index_space_domain(ctx, part_is);
    assert(domain.get_dim() == NDIM);
  }
  // Current assume forward and grad share the same index space
  assert(tensor.region.get_index_space() == tensor.region_grad.get_index_space());
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, tensor.region.get_index_space());
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
      ctx, tensor.region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor.region, ip);
  part_bwd = runtime->get_logical_partition(ctx, tensor.region_grad, ip);
}

template<int NDIM, int TDIM>
void FFModel::create_data_parallel_partition_with_diff_dims(const Tensor& tensor,
                                                            const IndexSpaceT<TDIM>& part_is,
                                                            LogicalPartition& part_fwd,
                                                            LogicalPartition& part_bwd)
{
  assert(tensor.numDim == NDIM);
  // Current assume forward and grad share the same index space
  assert(tensor.region.get_index_space() == tensor.region_grad.get_index_space());
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, tensor.region.get_index_space());
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
      ctx, tensor.region.get_index_space(), part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  part_fwd = runtime->get_logical_partition(ctx, tensor.region, ip);
  part_bwd = runtime->get_logical_partition(ctx, tensor.region_grad, ip);
}

// This function assumes:
// 1. the outer most dim of weight is channel out
// 2. partition is 2D (sample, channel_out)
template<int NDIM, int TDIM>
Parameter FFModel::create_linear_weight(Op* op,
                                        const int dims[],
                                        DataType data_type,
                                        Initializer* initializer,
                                        bool create_grad,
                                        Parameter::CommType comm_type)
{
  std::string pcname = op->name;
  IndexSpaceT<TDIM> part_is = (IndexSpaceT<TDIM>)get_or_create_task_is(TDIM, pcname);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++)
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  Parameter weight;
  weight.type = comm_type;
  weight.owner_op = op;
  weight.numDim = NDIM;
  weight.data_type = data_type;
  for (int i = 0; i < NDIM; i++)
    weight.adim[i] = dims[NDIM-1-i];
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
  // Step 1: forward region and partition
  if (weight.type == Parameter::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region = runtime->create_logical_region(ctx, is, fs);
    assert(dims[0] % num_parts[0] == 0);
    hi[NDIM-1] = dims[0] / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0] / num_parts[0];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight.part = runtime->get_logical_partition(
        ctx, weight.region, ip);
  } else if (weight.type == Parameter::NCCL) {
    // Currently only support the sample dimension for operators with NCCL
    //ParallelConfig pconfig;
    //config.find_parallel_config(TDIM, pcname, pconfig);
    //assert(pconfig.is_data_parallel());
    for (int i = 0; i < TDIM-1; i++)
      assert(num_parts[i] == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++)
      num_batches *= num_parts[i];
    hi[NDIM-1] = num_batches * dims[0] -1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = dims[0] / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0] / num_parts[0];
    for (int i = 1; i < TDIM; i++)
      transform[NDIM-1][i] = transform[NDIM-1][i-1] * num_parts[i-1];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight.part = runtime->get_logical_partition(
        ctx, weight.region, ip);
  } else {
    assert(false);
  }
  // Step 2: initialize region
  if (initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    initializer->init(this, &weight);
  }
  // Step 3: backward region
  if (create_grad) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    int num_batches = 1;
    for (int i = 1; i < TDIM; i++)
      num_batches *= num_parts[i];
    hi[NDIM-1] = num_batches * dims[0] -1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = dims[0] / num_parts[0] - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, TDIM> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < TDIM; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0] / num_parts[0];
    for (int i = 1; i < TDIM; i++)
      transform[NDIM-1][i] = transform[NDIM-1][i-1] * num_parts[i-1];
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight.part_grad = runtime->get_logical_partition(
        ctx, weight.region_grad, ip);
  }
  return weight;
}

template<int NDIM>
Parameter FFModel::create_conv_weight(Op* op,
                                      const int dims[],
                                      DataType data_type,
                                      Initializer* initializer,
                                      bool create_grad,
                                      Parameter::CommType comm_type)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  std::string pcname = op->name;
  IndexSpaceT<4> part_is = (IndexSpaceT<4>) get_or_create_task_is(4, pcname);
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, part_is);
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  // Currently assume we do not split over the channel dimension
  assert(num_par_c == 1);
  Parameter weight;
  weight.type = comm_type;
  weight.owner_op = op;
  weight.numDim = NDIM;
  weight.data_type = data_type;
  for (int i = 0; i < NDIM; i++)
    weight.adim[i] = dims[NDIM-1-i];
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
  // Step 1: forward region and partition
  if (weight.type == Parameter::PS) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region = runtime->create_logical_region(ctx, is, fs);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 4; j++)
        transform[i][j] = 0;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, rect);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight.part = runtime->get_logical_partition(
        ctx, weight.region, ip);
  } else if (weight.type == Parameter::NCCL) {
    // Currently only support sample and attribute parallelism for NCCL communication
    assert(num_par_c == 1);
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    hi[NDIM-1] = num_par_n * num_par_h * num_par_w * dims[0] - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = dims[0]-1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 4; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0];
    transform[NDIM-1][1] = dims[0] * num_par_w;
    transform[NDIM-1][2] = dims[0] * num_par_w * num_par_h;
    transform[NDIM-1][3] = dims[0] * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight.part = runtime->get_logical_partition(
        ctx, weight.region, ip);
  } else {
    // Unsupported Parameter type
    assert(false);
  }
  // Step 2: initialize region
  if (initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    initializer->init(this, &weight);
  }
  // Step 3: backwar regin and partition
  if (create_grad) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    hi[NDIM-1] = num_par_n * num_par_h * num_par_w * dims[0] - 1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = dims[0]-1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 4> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 4; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0];
    transform[NDIM-1][1] = dims[0] * num_par_w;
    transform[NDIM-1][2] = dims[0] * num_par_w * num_par_h;
    transform[NDIM-1][3] = dims[0] * num_par_w * num_par_h * num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    weight.part_grad = runtime->get_logical_partition(
        ctx, weight.region_grad, ip);
  }
  return weight;
}

template<int NDIM, int TDIM>
Tensor FFModel::create_linear_replica(const int dims[],
                                      const IndexSpaceT<TDIM>& task_is,
                                      DataType data_type)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  assert(NDIM >= 2);
  Rect<TDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_parts[TDIM];
  for (int i = 0; i < TDIM; i++)
    num_parts[i] = part_rect.hi[i] - part_rect.lo[i] + 1;
  Tensor replica;
  replica.numDim = NDIM;
  replica.data_type = data_type;
  for (int i = 0; i < NDIM; i++)
    replica.adim[i] = dims[NDIM-1-i];
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
  replica.region_grad = runtime->create_logical_region(ctx, is, fs);
  assert(dims[0] == num_parts[0]);
  //assert(dims[1] % num_parts[1] == 0);
  hi[NDIM-1] = dims[0] / num_parts[0] - 1;
  //hi[NDIM-2] = dims[1] / num_parts[1] - 1;
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
  Transform<NDIM, TDIM> transform;
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < TDIM; j++)
      transform[i][j] = 0;
  transform[NDIM-1][0] = 1;
  //transform[NDIM-2][1] = dims[1] / num_parts[1];
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  replica.part_grad = runtime->get_logical_partition(
    ctx, replica.region_grad, ip);
  return replica;
}

IndexSpace FFModel::get_task_is(ParallelConfig pc) const
{
  std::map<ParallelConfig, IndexSpace, ParaConfigCompare>::const_iterator iter;
  iter = taskIs.find(pc);
  assert(iter != taskIs.end());
  return iter->second;
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

void FFModel::forward()
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->forward(*this);
}

void FFModel::compute_metrics()
{
  Op* final_layer = layers[layers.size()-1];
  assert(final_layer->numOutputs == 1);
  metrics_op->compute(this, &(final_layer->outputs[0]), &label_tensor);
}

void FFModel::backward()
{
  // Compute metrics
  Op* final_layer = layers[layers.size()-1];
  assert(final_layer->numOutputs == 1);
  metrics_op->compute(this, &(final_layer->outputs[0]), &label_tensor);
  // Compute the gradients of the final layer wrt loss
  loss_op->backward(this, &(final_layer->outputs[0]), &label_tensor);
  // Perform backpropagation
  // std::set<LogicalRegion> resetedInputGrads;
  for (int l = layers.size() - 1; l >= 0; l--) {
#ifdef ENABLE_RESNET_INPUT_GRADIENT_OPTIMIZATION
    for (int i = 0; i < layers[l]->numInputs; i++)
      if (resetedInputGrads.find(layers[l]->inputs[i].region) == resetedInputGrads.end()) {
        resetedInputGrads.insert(layers[l]->inputs[i].region);
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
    optimizer->update(&(parameters[i]));
  }
}

void FFModel::compile(Optimizer* _optimizer,
                      LossType loss_type,
                      const std::vector<MetricsType>& metrics)
{
  optimizer = _optimizer;
  compile(loss_type, metrics);
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
          if (opl->inputs[idx].owner_op == layers[i]) {
            assert(!found);
            found = true;
            if (i > start) start = i;
          }
        assert(found || (opl->inputs[idx].owner_op == NULL));
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
              if ((op->inputs[idx].owner_op == layers[l])
              || (op->inputs[idx].owner_op == layers[i]))
              {
                int found = -1;
                for (int k = 0; k < fused_op->numOutputs; k++)
                  if (fused_op->outputs[k].region == op->inputs[idx].region) {
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
                      const std::vector<MetricsType>& metrics)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
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

  for (size_t l = 0; l < layers.size(); l++) {
    Op* op = layers[l];
    for (int i = 0; i < op->numInputs; i++) {
      if (op->inputs[i].owner_op == NULL) {
        // User created tensor
        op->inputs[i] = op->inputs[i];
      } else {
        // Refresh op's input tensor
        int tsIdx = op->inputs[i].owner_idx;
        op->inputs[i] = op->inputs[i].owner_op->outputs[tsIdx];
      }
    }
    op->create_output_and_partition(*this);
    op->create_weights(*this);
    for (int i = 0; i < op->numWeights; i++) {
      parameters.push_back(op->weights[i]);
    }
  }

  // Check correctness
  for (size_t l = 0; l < layers.size(); l++) {
    Op* op = layers[l];
    for (int i = 0; i < op->numOutputs; i++) {
      assert(op->outputs[i].owner_op == op);
      assert(op->outputs[i].owner_idx == i);
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
            if (new_layers[i]->inputs[idx].owner_op == new_layers[j])
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
              assert(fused->inputs[my_off].region == old_op->inputs[i].region);
            } else if (fused->op_input_source[i+ioff] == FusedOp::SOURCE_OUTPUT) {
              assert(fused->outputs[my_off].region == old_op->inputs[i].region);
            } else
              assert(false);
          }
          for (int i = 0; i < fused->op_num_weights[op]; i++) {
            int my_off = fused->op_weight_idx[i+woff];
            assert(fused->op_weight_source[i+woff] == FusedOp::SOURCE_WEIGHT);
            assert(fused->weights[my_off].region == old_op->weights[i].region);
          }
          for (int i = 0; i < fused->op_num_outputs[op]; i++) {
            int my_off = fused->op_output_idx[i+ooff];
            assert(fused->op_output_source[i+ooff] == FusedOp::SOURCE_OUTPUT);
            assert(fused->outputs[my_off].region == old_op->outputs[i].region);
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
  }
  Op* final_layer = layers[layers.size()-1];
  // FIXME: currently assume the final layer has exactly one output
  assert(final_layer->numOutputs == 1);

  for (size_t i = 0; i < layers.size(); i++) {
      Op* op = layers[i];
      printf("layer[%zu]: type(%d)\n", i, layers[i]->op_type);
      for (int j = 0; j < op->numInputs; j++) {
        LogicalRegion handle = op->inputs[j].region;
        printf("inputs[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                          handle.get_field_space().get_id(),
                          handle.get_tree_id());
      }
      for (int j = 0; j < op->numOutputs; j++) {
        LogicalRegion handle = op->outputs[j].region;
        printf("outputs[%d] region(%d,%d,%d)\n", j, handle.get_index_space().get_id(),
                          handle.get_field_space().get_id(),
                          handle.get_tree_id());
      }
  }
  //assert(final_layer->outputs[0].numDim == 2);
  int dims[MAX_TENSOR_DIM], num_dims;
  num_dims = final_layer->outputs[0].numDim;
  // Note that FlexFlow's runtim internally reverse the array ordering
  for (int i = 0; i < num_dims; i++)
    dims[i] = final_layer->outputs[0].adim[num_dims-1-i];
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
}

void FFModel::rewrite(const std::map<Op*, ParallelConfig>& current,
                      std::map<Op*, ParallelConfig>& next) const
{
  next = current;
  size_t opId = std::rand() % layers.size();
  //TODO: need to make sure opId is not an output layer of the model
  if (opId == layers.size() - 1)
    return;
  next[layers[opId]] = layers[opId]->get_random_parallel_config(*this);
}

void FFModel::optimize(Simulator* simulator,
                       std::map<Op*, ParallelConfig>& best,
                       size_t budget, float alpha) const
{
  // Start from data parallel
  std::map<Op*, ParallelConfig> current, next;
  float best_runtime = simulator->simulate_runtime(this, best);
  current = best;
  float current_runtime = best_runtime;
  for (size_t iter = 0; iter < budget; iter++) {
    rewrite(current, next);
    float next_runtime = simulator->simulate_runtime(this, next);
    if (iter % 100 == 0) {
      printf("iter(%zu) cur(%.2lf) next(%.2lf) best(%.2lf)\n", iter,
             current_runtime, next_runtime, best_runtime);
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
  simulator->simulate_runtime(this, best, this->config.export_strategy_task_graph_file);
  std::map<Op*, ParallelConfig>::const_iterator it;
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
#ifdef DEADCODE
  ArgumentMap arg_map;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  for (size_t p = 0; p < parameters.size(); p++) {
    Domain domain = runtime->get_index_partition_color_space(
        ctx, parameters[p].part_grad.get_index_partition());
    IndexSpace task_is = get_or_create_task_is(domain);
    IndexLauncher launcher(ZERO_INIT_TASK_ID, task_is,
                           TaskArgument(NULL, 0), arg_map,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string(parameters[p].pcname)));
    launcher.add_region_requirement(
        RegionRequirement(parameters[p].part_grad, 0/*projection*/,
                          WRITE_ONLY, EXCLUSIVE, parameters[p].region_grad));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
#endif
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
    default: assert(false && "Not supported Operator type"); return "Unsupported";
  }
}

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
// class FFConfig
// ========================================================

// Default Config Parameters
struct DefaultConfig {
  const static int epochs = 1;
  const static int iterations = 1;
  const static int batchSize = 64;
  const static bool profiling = false;
  const static bool debug = false;
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
};

FFConfig::FFConfig()
{
  epochs = DefaultConfig::epochs;
  iterations = DefaultConfig::iterations;
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
  enable_sample_parallel = DefaultConfig::enableSampleParallel;
  enable_parameter_parallel = DefaultConfig::enableParameterParallel;
  enable_attribute_parallel = DefaultConfig::enableAttributeParallel;

  import_strategy_file = "";
  export_strategy_file = "";
  export_strategy_task_graph_file = "";
  dataset_path = "";
  syntheticInput = false;
  perform_fusion = false;
}

void FFConfig::parse_args(char **argv, int argc)
{
  for (int i = 1; i < argc; i++)
  {
    if ((!strcmp(argv[i], "-e")) || (!strcmp(argv[i], "--epochs"))) {
      epochs = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-i")) || (!strcmp(argv[i], "--iterations"))) {
      iterations = atoi(argv[++i]);
      continue;
    }
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
  }
}

// Perform data parallelsim across machines
class DataParallelShardingFunctor : public ShardingFunctor {
public:
  DataParallelShardingFunctor(void);
  ~DataParallelShardingFunctor(void);
public:
  ShardID shard(const DomainPoint &point,
                const Domain &full_space,
                const size_t total_shards);
};

DataParallelShardingFunctor::DataParallelShardingFunctor(void)
: ShardingFunctor() {}

DataParallelShardingFunctor::~DataParallelShardingFunctor(void)
{}

ShardID DataParallelShardingFunctor::shard(const DomainPoint &point,
                                           const Domain &full_space,
                                           const size_t total_shards)
{
  assert(point.get_dim() == full_space.get_dim());
  int idx = full_space.get_dim() - 1;
  int samples = full_space.hi()[idx] - full_space.lo()[idx] + 1;
  int samples_per_shard = (samples + total_shards - 1) / total_shards;
  return (point[idx] - full_space.lo()[idx]) / samples_per_shard;
}

void register_internal_tasks()
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
    TaskVariantRegistrar registrar(BATCHNORM_INIT_PARA_TASK_ID, "bm_init_para_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<BatchNorm::init_para_task>(
        registrar, "bm_init_para_task");
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
#ifdef FF_ENABLE_NCCL
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
  // Search
  {
    TaskVariantRegistrar registrar(STRATEGY_SEARCH_TASK_ID,
                                   "Stretegy Search");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Simulator::strategy_search_task>(
        registrar, "Stretegy Search Task");
  }
  // DUMMY task
  {
    TaskVariantRegistrar registrar(DUMMY_TASK_ID, "dummy_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<UtilityTasks::dummy_task>(registrar, "dummy_task");
  }
}

#if !defined(FF_USE_PYTHON)
// ========================================================
// Task and mapper registrations
// ========================================================
int main(int argc, char** argv)
{
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  register_internal_tasks();

  // Register custom tasks
  register_custom_tasks();

  // Init MPI
#if defined(GASNET_CONDUIT_MPI) || defined(REALM_USE_MPI)
  // The GASNet MPI conduit and/or the Realm MPI network layer
  // require that MPI be initialized for multiple threads
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  // If you fail this assertion, then your version of MPI
  // does not support calls from multiple threads and you
  // cannot use the GASNet MPI conduit
  if (provided < MPI_THREAD_MULTIPLE)
    printf("ERROR: Your implementation of MPI does not support "
           "MPI_THREAD_MULTIPLE which is required for use of the "
           "GASNet MPI conduit or the Realm MPI network layer "
           "with the Legion-MPI Interop!\n");
  assert(provided == MPI_THREAD_MULTIPLE);
#elif defined(FF_ENABLE_NCCL)
  // Perform MPI start-up like normal for most GASNet conduits
  MPI_Init(&argc, &argv);
#endif

  DataParallelShardingFunctor* sharding_functor = new DataParallelShardingFunctor();
  Runtime::preregister_sharding_functor(DataParallelShardingID, sharding_functor);

  Runtime::add_registration_callback(update_mappers);
  return Runtime::start(argc, argv);
}

#else
void register_flexflow_tasks()
{
  register_internal_tasks();

  register_c_custom_tasks();

  DataParallelShardingFunctor* sharding_functor = new DataParallelShardingFunctor();
  Runtime::preregister_sharding_functor(DataParallelShardingID, sharding_functor);
}

#endif // FF_USE_PYTHON

// template instantiations
#define DIMFUNC(DIM) \
  template Tensor FFModel::create_tensor<DIM>(const int* dims, DataType data_type, const Op* owner_op, bool create_grad); \
  template Tensor FFModel::create_constant<DIM>(const int* dims, float value, DataType data_type); \
  template void FFModel::create_disjoint_partition<DIM>(const Tensor& tensor, const IndexSpaceT<DIM>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
  LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC

#define DIMFUNC(D1,D2) \
  template void FFModel::create_data_parallel_partition_with_diff_dims<D1, D2>(const Tensor& tensor, const IndexSpaceT<D2>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
  LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

template Parameter FFModel::create_conv_weight<4>(Op* op, const int* dims, DataType data_type, Initializer* initializer, bool create_grad, Parameter::CommType comm_type);
template Parameter FFModel::create_conv_weight<1>(Op* op, const int* dims, DataType data_type, Initializer* initializer, bool create_grad, Parameter::CommType comm_type);

#define DIMFUNC(D1,D2) \
  template Parameter FFModel::create_linear_weight<D1, D2>(Op* op, const int* dims, DataType data_type, Initializer* initializer, bool create_grad, Parameter::CommType comm_type);
  LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

#define DIMFUNC(D1,D2) \
  template Tensor FFModel::create_linear_replica<D1>(const int* dims, const IndexSpaceT<D2>& part_is, DataType data_type);
  LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC

template float* Tensor::get_raw_ptr<float>(FFConfig &config);
template int32_t* Tensor::get_raw_ptr<int32_t>(FFConfig &config);
