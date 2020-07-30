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

Op::Op(FFModel& model,
       const std::string& _name,
       const Tensor& _input)
: numInputs(1), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  inputs[0] = _input;
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
}

Op::Op(FFModel& model,
       const std::string& _name,
       const Tensor& _input1,
       const Tensor& _input2)
: numInputs(2), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  inputs[0] = _input1;
  inputs[1] = _input2;
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
}

Op::Op(FFModel& model,
       const std::string& _name,
       int n, const Tensor* _inputs)
: numInputs(n), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  assert(n <= MAX_NUM_INPUTS);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_type = inputs[0].data_type;
  }
}

Op::Op(FFModel& model,
       const std::string& _name,
       int _numInputs)
: numInputs(_numInputs), numWeights(0), numOutputs(1)
{
  std::string pcname = _name + "_" + std::to_string(model.op_global_guid++);
  assert(pcname.length() < MAX_OPNAME);
  std::strcpy(name, pcname.c_str());
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
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
    launcher.add_field(i + numWeights, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

FFModel::FFModel(FFConfig& _config)
: op_global_guid(100), config(_config)
{
  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
  // Load strategy file
  if (config.strategyFile == "") {
  } else {
    load_strategies_from_file(config.strategyFile, config.strategies);
  }
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
#ifdef DEADCODE
  // Build logical regions for images
  Rect<4> part_rect(Point<4>(0, 0, 0, 0),
      Point<4>(0, 0, 0, config.numNodes * config.workersPerNode-1));
  IndexSpaceT<4> part_is = runtime->create_index_space(ctx, part_rect);
  Rect<4> image_rect(Point<4>(0, 0, 0, 0),
    Point<4>(config.inputWidth-1, config.inputHeight-1, 2, config.batchSize-1));
  IndexSpaceT<4> image_is = runtime->create_index_space(ctx, image_rect);
  LogicalRegion image_lr = runtime->create_logical_region(ctx, image_is,
                               config.field_space);
  //LogicalRegion image_grad_lr = runtime->create_logical_region(ctx, image_is,
  //                                  config.field_space);
  int extentW = config.inputWidth;
  int extentH = config.inputHeight;
  int extentC = 3;
  assert(config.batchSize % (config.numNodes * config.workersPerNode) == 0);
  int extentN = config.batchSize / (config.numNodes * config.workersPerNode);
  Rect<4> extent(Point<4>(0, 0, 0, 0),
                 Point<4>(extentW-1, extentH-1, extentC-1, extentN-1));
  Transform<4, 4> transform;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      transform[i][j] = 0;
  transform[0][0] = extentW;
  transform[1][1] = extentH;
  transform[2][2] = 3;
  transform[3][3] = extentN;
  IndexPartition image_ip =
    runtime->create_partition_by_restriction(ctx, image_is, part_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, image_ip));
  assert(runtime->is_index_partition_complete(ctx, image_ip));
  LogicalPartition image_lp = runtime->get_logical_partition(ctx, image_lr, image_ip);
  //LogicalPartition image_grad_lp =
  //  runtime->get_logical_partition(ctx, image_grad_lr, image_ip);
  inputImage.numDim = 4;
  inputImage.adim[0] = config.inputWidth;
  inputImage.adim[1] = config.inputHeight;
  inputImage.adim[2] = 3;
  inputImage.adim[3] = config.batchSize;
  inputImage.pdim[0] = extentW;
  inputImage.pdim[1] = extentH;
  inputImage.pdim[2] = 3;
  inputImage.pdim[3] = extentN;
  inputImage.region = image_lr;
  inputImage.region_grad = LogicalRegion::NO_REGION;
  inputImage.part = image_lp;
  inputImage.part_grad = LogicalPartition::NO_PART;
  // Build local regions for input raw images
  int extentHWC = config.inputHeight * config.inputWidth * 3;
  Rect<2> raw_rect(Point<2>(0, 0), Point<2>(extentHWC-1, config.batchSize-1));
  IndexSpaceT<2> raw_is = runtime->create_index_space(ctx, raw_rect);
  LogicalRegion raw_lr =
      runtime->create_logical_region(ctx, raw_is, config.field_space);
  Transform<2, 4> raw_trans;
  Rect<2> raw_ext(Point<2>(0, 0), Point<2>(extentHWC-1, extentN-1));
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      raw_trans[i][j] = 0;
  raw_trans[1][3] = extentN;
  IndexPartition raw_ip =
    runtime->create_partition_by_restriction(ctx, raw_is, part_is, raw_trans, raw_ext);
  assert(runtime->is_index_partition_disjoint(ctx, raw_ip));
  assert(runtime->is_index_partition_complete(ctx, raw_ip));
  LogicalPartition raw_lp = runtime->get_logical_partition(ctx, raw_lr, raw_ip);
  inputRaw.numDim = 2; //Dim [HWC, N]
  inputRaw.adim[0] = extentHWC;
  inputRaw.adim[1] = config.batchSize;
  inputRaw.pdim[0] = extentHWC;
  inputRaw.pdim[1] = extentN;
  inputRaw.region = raw_lr;
  inputRaw.part = raw_lp;
#endif
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
    tensor.pdim[i] = extent.hi[i] - extent.lo[i] + 1;
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
  std::set<LogicalRegion> resetedInputGrads;
  for (int l = layers.size() - 1; l >= 0; l--) {
    for (int i = 0; i < layers[l]->numInputs; i++)
      if (resetedInputGrads.find(layers[l]->inputs[i].region) == resetedInputGrads.end()) {
        resetedInputGrads.insert(layers[l]->inputs[i].region);
      } else {
        // This input's gradients has been reseted by other layers
        // So we should not do it again
        layers[l]->resetInputGrads[i] = false;
      }
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

void FFModel::compile()
{
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
  printf("in update_metrics_task\n");
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
  const static int inputHeight = 224;
  const static int inputWidth = 224;
  const static bool profiling = false;
  constexpr static float learningRate = 0.01f;
  constexpr static float weightDecay = 0.0001f;
  const static size_t workSpaceSize = (size_t)1 * 1024 * 1024 * 1024; // 2GB
  const static int numNodes = 1;
  const static int workersPerNode = 0;
  const static int loadersPerNode = 4;
};

FFConfig::FFConfig()
{
  epochs = DefaultConfig::epochs;
  iterations = DefaultConfig::iterations;
  batchSize = DefaultConfig::batchSize;
  inputHeight = DefaultConfig::inputHeight;
  inputWidth = DefaultConfig::inputWidth;
  profiling = DefaultConfig::profiling;
  learningRate = DefaultConfig::learningRate;
  weightDecay = DefaultConfig::weightDecay;
  workSpaceSize = DefaultConfig::workSpaceSize;
  numNodes = DefaultConfig::numNodes;
  loadersPerNode = DefaultConfig::loadersPerNode;
  workersPerNode = DefaultConfig::workersPerNode;
  strategyFile = "";
  datasetPath = "";
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
      datasetPath = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-s")) || (!strcmp(argv[i], "--strategy"))) {
      strategyFile = std::string(argv[++i]);
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
  // Embedding task GPU
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
