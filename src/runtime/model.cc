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
#include "mapper.h"
#include "dirent.h"

using namespace std;

LegionRuntime::Logger::Category log_model("ff");

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
  if (pc.nDims != numDim)
    return false;
  for (int i = 0; i < numDim; i++)
    if (adim[i] % pc.dim[i] != 0)
      return false;
  tensor.numDim = numDim;
  for (int i = 0; i < numDim; i++)
    tensor.adim[i] = adim[i] / pc.dim[i];
  tensor.data_type = data_type;
  return true;
}

bool Tensor::get_output_sub_tensor(const ParallelConfig& pc,
                                   Tensor& tensor,
                                   OperatorType type)
{
  if (pc.nDims != numDim)
    return false;
  for (int i = 0; i < numDim; i++)
    if (adim[i] % pc.dim[i] != 0)
      return false;
  tensor.numDim = numDim;
  for (int i = 0; i < numDim; i++)
    tensor.adim[i] = adim[i] / pc.dim[i];
  tensor.data_type = data_type;
  return true;
}

size_t Tensor::get_volume()
{
  size_t volume = 1;
  for (int i = 0; i < numDim; i++)
    volume *= adim[i];
  return volume;
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const std::string& _name,
       const Tensor& _input)
: op_type(_op_type), numInputs(1), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
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
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const std::string& _name,
       const Tensor& _input1,
       const Tensor& _input2)
: op_type(_op_type), numInputs(2), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
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
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const std::string& _name,
       int n, const Tensor* _inputs)
: op_type(_op_type), numInputs(n), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
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
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
}

Op::Op(FFModel& model,
       OperatorType _op_type,
       const std::string& _name,
       int _numInputs)
: op_type(_op_type), numInputs(_numInputs), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  //for (int i = 0; i < numInputs; i++) {
  //  trainableInputs[i] = true;
  //  resetInputGrads[i] = true;
  //}
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
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
  for (int i = 0; i < d.dim; i++) {
    // Assume an equal partitioning
    assert(outputs[output_idx].adim[i] % pc.dim[i] == 0);
    int dim_size = outputs[output_idx].adim[i] / pc.dim[i];
    d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
    d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    part_idx = part_idx / pc.dim[i];
  }
  return d;
}

Domain Op::get_input_tensor_shape(const ParallelConfig& pc,
                                  int input_idx, int part_idx)
{
  assert(input_idx < numInputs);
  Domain d;
  d.dim = inputs[input_idx].numDim;
  for (int i = 0; i < d.dim; i++) {
    // Assume an equal partitioning
    assert(inputs[input_idx].adim[i] % pc.dim[i] == 0);
    int dim_size = inputs[input_idx].adim[i] / pc.dim[i];
    d.rect_data[i] = (part_idx % pc.dim[i]) * dim_size;
    d.rect_data[i + d.dim] = d.rect_data[i] + dim_size - 1;
    part_idx = part_idx / pc.dim[i];
  }
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

FFModel::FFModel(FFConfig& _config)
: op_global_guid(100), config(_config),
  optimizer(NULL), loss_op(NULL), metrics_op(NULL)
{
  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
  // Load strategy file
  for (int i = FFConfig::DataParallelism_1D; i <= FFConfig::DataParallelism_4D; i++) {
    ParallelConfig pc;
    pc.device_type = ParallelConfig::GPU;
    pc.nDims = i - FFConfig::DataParallelism_1D + 1;
    for (int j = 0; j < pc.nDims; j++)
      pc.dim[j] = 1;
    pc.dim[pc.nDims-1] = config.workersPerNode * config.numNodes;
    for (int j = 0; j < pc.dim[pc.nDims-1]; j++)
      pc.device_ids[j] = j;
    config.strategies[i] = pc;
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

  // Init performance metrics
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID, TaskArgument(NULL, 0));
  current_metrics = runtime->execute_task(ctx, launcher);

  // Init CUDA library on each worker
  ArgumentMap local_args;
  size_t workSpaceSize = config.workSpaceSize;
  Rect<2> task_rect(Point<2>(0, 0),
                    Point<2>(0, config.workersPerNode * config.numNodes - 1));
  IndexSpaceT<2> task_is = runtime->create_index_space(ctx, task_rect);
  IndexLauncher initLauncher(FF_INIT_TASK_ID, task_is,
                             TaskArgument(&workSpaceSize, sizeof(workSpaceSize)), local_args,
                             Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                             FFConfig::DataParallelism_2D);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<2> it(task_rect); it(); it++) {
    handlers[idx++] = fm.get_result<FFHandler>(*it);
  }
}

template<int NDIM>
Tensor FFModel::create_tensor(const int dims[],
                              const std::string& pc_name,
                              DataType data_type,
                              bool create_grad)
{
  ParallelConfig pc;
  assert(config.find_parallel_config(NDIM, pc_name, pc));
  IndexSpaceT<NDIM> task_is = IndexSpaceT<NDIM>(get_or_create_task_is(pc));
  return create_tensor(dims, task_is, data_type, create_grad);
}

template<int NDIM>
Tensor FFModel::create_constant(const int dims[],
                                const std::string& pc_name,
                                float value,
                                DataType data_type)
{
  // constant created in this way is not part of any operator
  // so we assume it does not have gradients
  Tensor tensor = create_tensor<NDIM>(dims, pc_name, data_type, false/*create_grad*/);
  ConstantInitializer initializer(value);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  initializer.init(ctx, runtime, &tensor);
  return tensor;
}

template<int NDIM>
Tensor FFModel::create_tensor(const int dims[],
                              const IndexSpaceT<NDIM>& part_is,
                              DataType data_type,
                              bool create_grad)
{
  Tensor tensor;
  tensor.data_type = data_type;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
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

  return tensor;
}

template<int NDIM>
void FFModel::create_disjoint_partition(const Tensor& tensor,
                                        const IndexSpaceT<NDIM>& part_is,
                                        LogicalPartition& part_fwd,
                                        LogicalPartition& part_bwd)
{
  assert(tensor.numDim == NDIM);
  // Current assume forward and grad share the same index space
  assert(tensor.region.get_index_space() == tensor.region_grad.get_index_space());
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
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
template<int NDIM>
Parameter FFModel::create_linear_weight(Op* op,
                                        const int dims[],
                                        const IndexSpaceT<2>& part_is,
                                        DataType data_type,
                                        Initializer* initializer,
                                        bool create_grad)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, part_is);
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  Parameter weight;
  weight.pcname = op->name;
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
  {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region = runtime->create_logical_region(ctx, is, fs);
    assert(dims[0] % num_par_c == 0);
    hi[NDIM-1] = dims[0] / num_par_c - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 2> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 2; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0] / num_par_c;
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, part_is, transform, extent);
    assert(runtime->is_index_partition_complete(ctx, ip));
    weight.part = runtime->get_logical_partition(
        ctx, weight.region, ip);
  }
  // Step 2: initialize region
  if (initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    initializer->init(ctx, runtime, &weight);
  }
  // Step 3: backward region
  if (create_grad) {
    Point<NDIM> hi;
    for (int i = 0; i < NDIM; i++)
      hi[i] = dims[NDIM-1-i]-1;
    hi[NDIM-1] = num_par_n * dims[0] -1;
    Rect<NDIM> rect(Point<NDIM>::ZEROES(), hi);
    IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
    weight.region_grad = runtime->create_logical_region(ctx, is, fs);
    hi[NDIM-1] = dims[0] / num_par_c - 1;
    Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
    Transform<NDIM, 2> transform;
    for (int i = 0; i < NDIM; i++)
      for (int j = 0; j < 2; j++)
        transform[i][j] = 0;
    transform[NDIM-1][0] = dims[0] / num_par_c;
    transform[NDIM-1][1] = dims[0];
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
                                      const IndexSpaceT<4>& part_is,
                                      DataType data_type,
                                      Initializer* initializer,
                                      bool create_grad)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, part_is);
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  // Currently assume we do not split over the channel dimension
  assert(num_par_c == 1);
  Parameter weight;
  weight.pcname = op->name;
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
  {
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
  }
  // Step 2: initialize region
  if (initializer == NULL) {
    assert(false); // add weight initializer should be set before
  } else {
    initializer->init(ctx, runtime, &weight);
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

template<int NDIM>
Tensor FFModel::create_linear_replica(const int dims[],
                                      const IndexSpaceT<2>& task_is,
                                      DataType data_type)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  assert(NDIM >= 2);
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
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
  assert(dims[0] == num_par_c);
  assert(dims[1] % num_par_n == 0);
  hi[NDIM-1] = dims[0] / num_par_c - 1;
  hi[NDIM-2] = dims[1] / num_par_n - 1;
  Rect<NDIM> extent(Point<NDIM>::ZEROES(), hi);
  Transform<NDIM, 2> transform;
  for (int i = 0; i < NDIM; i++)
    for (int j = 0; j < 2; j++)
      transform[i][j] = 0;
  transform[NDIM-1][0] = 1;
  transform[NDIM-2][1] = dims[1] / num_par_n;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  replica.part_grad = runtime->get_logical_partition(
    ctx, replica.region_grad, ip);
  return replica;
}

IndexSpace FFModel::get_or_create_task_is(ParallelConfig pc)
{
  if (taskIs.find(pc) != taskIs.end())
    return taskIs[pc];
  IndexSpace task_is;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  switch (pc.nDims) {
    case 1:
    {
      Rect<1> task_rect(Point<1>(0), Point<1>(pc.dim[0]-1));
      task_is = runtime->create_index_space(ctx, task_rect);
      break;
    } 
    case 2:
    {
      Rect<2> task_rect(Point<2>(0, 0), Point<2>(pc.dim[0]-1, pc.dim[1]-1));
      task_is = runtime->create_index_space(ctx, task_rect);
      break;
    }
    case 3:
    {
      Rect<3> task_rect(Point<3>(0, 0, 0),
                        Point<3>(pc.dim[0]-1, pc.dim[1]-1, pc.dim[2]-1));
      task_is = runtime->create_index_space(ctx, task_rect);
      break;
    }
    case 4:
    {
      Rect<4> task_rect(Point<4>(0, 0, 0, 0),
                        Point<4>(pc.dim[0]-1, pc.dim[1]-1, pc.dim[2]-1, pc.dim[3]-1));
      task_is = runtime->create_index_space(ctx, task_rect);
      break;
    }
    default:
      assert(false);
  }
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
  TaskLauncher launcher(UPDATE_METRICS_TASK_ID, TaskArgument(NULL, 0));
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

void FFModel::compile(LossType loss_type,
                      const std::vector<MetricsType>& metrics)
{
  if (config.import_strategy_file.length() > 0) {
    load_strategies_from_file(config.import_strategy_file, config.strategies);
  } else if (config.search_budget > 0) {
    // Launch the search task
    Context ctx = config.lg_ctx;
    Runtime* runtime = config.lg_hlr;
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
  Op* final_layer = layers[layers.size()-1];
  // FIXME: currently assume the final layer has exactly one output
  assert(final_layer->numOutputs == 1);
  // FIXME: currently assume the logit is 2D
  assert(final_layer->outputs[0].numDim == 2);
  int batch_size = final_layer->outputs[0].adim[1];
  int channel = final_layer->outputs[0].adim[0];
  DataType label_type = DT_FLOAT;
  if (loss_type == LOSS_SPARSE_CATEGORICAL_CROSSENTROPY) {
    // assign channel = 1 for sparse categorical labels
    channel = 1;
    label_type = DT_INT32;
  }
  // create label tensor
  {
    // Note that FlexFlow's runtim internally reverse the array ordering
    const int dims[] = {batch_size, channel};
    label_tensor = create_tensor<2>(dims, "", label_type);
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
  next[layers[opId]] = layers[opId]->get_random_parallel_config(*this);
}

void FFModel::optimize(Simulator* simulator,
                       std::map<Op*, ParallelConfig>& best,
                       size_t budget, float alpha) const
{
  // Start from data parallel
  std::map<Op*, ParallelConfig> current, next;
  for (size_t l = 0; l < layers.size(); l++) {
    current[layers[l]] = layers[l]->get_data_parallel_config(*this);
  }
  float best_runtime = simulator->simulate_runtime(this, current);
  best = current;
  float current_runtime = best_runtime;
  for (size_t iter = 0; iter < budget; iter++) {
    rewrite(current, next);
    float next_runtime = simulator->simulate_runtime(this, next);
    if (iter % 1 == 0) {
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

PerfMetrics FFModel::update_metrics_task(const Task *task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx, Runtime* runtime)
{
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
  all_metrics.print();
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
  constexpr static float learningRate = 0.01f;
  constexpr static float weightDecay = 0.0001f;
  const static size_t workSpaceSize = (size_t)1 * 1024 * 1024 * 1024; // 2GB
  const static int numNodes = 1;
  const static int workersPerNode = 0;
  const static int loadersPerNode = 4;
  const static size_t searchBudget = 0;
  const static size_t simulatorWorkSpaceSize = (size_t)2 * 1024 * 1024 * 1024; //2GB
  constexpr static float searchAlpha = 1.0f;
  const static bool searchOverlapBackwardUpdate = false;
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
  loadersPerNode = DefaultConfig::loadersPerNode;
  workersPerNode = DefaultConfig::workersPerNode;
  simulator_work_space_size = DefaultConfig::simulatorWorkSpaceSize;
  search_budget = DefaultConfig::searchBudget;
  search_alpha = DefaultConfig::searchAlpha;
  search_overlap_backward_update = DefaultConfig::searchOverlapBackwardUpdate;
  import_strategy_file = "";
  export_strategy_file = "";
  dataset_path = "";
  syntheticInput = false;
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
    if ((!strcmp(argv[i], "-import")) || (!strcmp(argv[i], "--import-strategy"))) {
      import_strategy_file = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-export")) || (!strcmp(argv[i], "--export-strategy"))) {
      export_strategy_file = std::string(argv[++i]);
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
      loadersPerNode = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--profiling"))
    {
      profiling = true;
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
    Runtime::preregister_task_variant<ElementBinary::init_task>(
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
  //{
  //  TaskVariantRegistrar registrar(CONV2D_INIT_PARA_TASK_ID, "Conv2D Init Para");
  //  registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
  //  registrar.set_leaf();
  //  Runtime::preregister_task_variant<Conv2D::init_para_task>(
  //      registrar, "Conv2D Init Para Task");
  //}
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
  {
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
    TaskVariantRegistrar registrar(CONCAT_INIT_TASK_ID, "concat_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<OpMeta*, Concat::init_task>(
        registrar, "concat_init_task");
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
  // Optimizer
  {
    TaskVariantRegistrar registrar(SGD_UPD_TASK_ID,
                                   "SGD Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SGDOptimizer::update_task>(
        registrar, "SGD Update Task");
  }
  {
    TaskVariantRegistrar registrar(ADAM_UPD_TASK_ID,
                                   "Adam Update");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<AdamOptimizer::update_task>(
        registrar, "Adam Update Task");
  }
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
template Tensor FFModel::create_tensor<1>(const int* dims, const std::string& pcname, DataType data_type, bool create_grad);
template Tensor FFModel::create_tensor<2>(const int* dims, const std::string& pcname, DataType data_type, bool create_grad);
template Tensor FFModel::create_tensor<3>(const int* dims, const std::string& pcname, DataType data_type, bool create_grad);
template Tensor FFModel::create_tensor<4>(const int* dims, const std::string& pcname, DataType data_type, bool create_grad);
template Tensor FFModel::create_constant<1>(const int* dims, const std::string & pcname, float value, DataType data_type);
template Tensor FFModel::create_constant<2>(const int* dims, const std::string & pcname, float value, DataType data_type);
template Tensor FFModel::create_constant<3>(const int* dims, const std::string & pcname, float value, DataType data_type);
template Tensor FFModel::create_constant<4>(const int* dims, const std::string & pcname, float value, DataType data_type);
template Tensor FFModel::create_tensor<1>(const int* dims, const IndexSpaceT<1>& part_is, DataType data_type, bool create_grad);
template Tensor FFModel::create_tensor<2>(const int* dims, const IndexSpaceT<2>& part_is, DataType data_type, bool create_grad);
template Tensor FFModel::create_tensor<3>(const int* dims, const IndexSpaceT<3>& part_is, DataType data_type, bool create_grad);
template Tensor FFModel::create_tensor<4>(const int* dims, const IndexSpaceT<4>& part_is, DataType data_type, bool create_grad);

template void FFModel::create_disjoint_partition<1>(const Tensor& tensor, const IndexSpaceT<1>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
template void FFModel::create_disjoint_partition<2>(const Tensor& tensor, const IndexSpaceT<2>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
template void FFModel::create_disjoint_partition<3>(const Tensor& tensor, const IndexSpaceT<3>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);
template void FFModel::create_disjoint_partition<4>(const Tensor& tensor, const IndexSpaceT<4>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);

template void FFModel::create_data_parallel_partition_with_diff_dims<4, 2>(const Tensor& tensor, const IndexSpaceT<2>& part_is, LogicalPartition& part_fwd, LogicalPartition& part_bwd);


template Parameter FFModel::create_conv_weight<4>(Op* op, const int* dims, const IndexSpaceT<4>& part_is, DataType data_type, Initializer* initializer, bool create_grad);
template Parameter FFModel::create_conv_weight<1>(Op* op, const int* dims, const IndexSpaceT<4>& part_is, DataType data_type, Initializer* initializer, bool create_grad);

template Parameter FFModel::create_linear_weight<2>(Op* op, const int* dims, const IndexSpaceT<2>& part_is, DataType data_type, Initializer* initializer, bool create_grad);
template Parameter FFModel::create_linear_weight<1>(Op* op, const int* dims, const IndexSpaceT<2>& part_is, DataType data_type, Initializer* initializer, bool create_grad);

template Tensor FFModel::create_linear_replica<3>(const int* dims, const IndexSpaceT<2>& part_is, DataType data_type);

template float* Tensor::get_raw_ptr<float>(FFConfig &config);
template int32_t* Tensor::get_raw_ptr<int32_t>(FFConfig &config);
