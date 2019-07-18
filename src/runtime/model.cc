/* Copyright 2018 Stanford
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
#include "dirent.h"

using namespace std;

LegionRuntime::Logger::Category log_model("ff");

Op::Op(const std::string& _name,
       const Tensor& _input)
: numLocals(0), numInputs(1)
{
  assert(_name.length() < MAX_OPNAME);
  std::strcpy(name, _name.c_str());
  inputs[0] = _input;
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
}

Op::Op(const std::string& _name,
       const Tensor& _input1,
       const Tensor& _input2)
: numLocals(0), numInputs(2)
{
  assert(_name.length() < MAX_OPNAME);
  std::strcpy(name, _name.c_str());
  inputs[0] = _input1;
  inputs[1] = _input2;
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
}

Op::Op(const std::string& _name,
       int n, const Tensor* _inputs)
: numLocals(0), numInputs(n)
{
  assert(_name.length() < MAX_OPNAME);
  std::strcpy(name, _name.c_str());
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  for (int i = 0; i < numInputs; i++) {
    trainableInputs[i] = true;
    resetInputGrads[i] = true;
  }
}

FFModel::FFModel(FFConfig& _config)
: config(_config)
{
  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
  // Create field space
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  // Build training dataset
  if (config.datasetPath.length() == 0) {
    dataLoader = NULL;
  } else {
    dataLoader = new DataLoader(config.datasetPath);
  }

  // Init CUDA library on each worker
  ArgumentMap local_args;
  size_t workSpaceSize = config.workSpaceSize;
  Rect<1> task_rect(Point<1>(0),
                    Point<1>(config.workersPerNode * config.numNodes - 1));
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);
  IndexLauncher initLauncher(FF_INIT_TASK_ID, task_is,
                             TaskArgument(&workSpaceSize, sizeof(workSpaceSize)),
                             local_args);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    handlers[*it] = fm.get_result<FFHandler>(*it);
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
Tensor FFModel::create_tensor(const int* dims,
                              const std::string& pc_name,
                              DataType data_type,
                              bool create_grad)
{
  assert(config.strategies.find(pc_name) != config.strategies.end());
  ParallelConfig pc = config.strategies[pc_name];
  IndexSpaceT<NDIM> task_is = IndexSpaceT<NDIM>(get_or_create_task_is(pc));
  return create_tensor(dims, task_is, data_type, create_grad);
}

template<int NDIM>
Tensor FFModel::create_tensor(const int* dims,
                              const IndexSpaceT<NDIM>& part_is,
                              DataType data_type,
                              bool create_grad)
{
  Tensor tensor;
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
      allocator.allocate_field(sizeof(int), FID_DATA);
      break;
    default:
      assert(false);
  }
  int points[NDIM];
  Point<NDIM> lo = Point<NDIM>::ZEROS();
  for (int i = 0; i < NDIM; i++)
    points[i] = dims[NDIM-1-i]-1;
  Point<NDIM> hi(points);
  Rect<NDIM> rect(lo, hi);
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  tensor.region = runtime->create_logical_region(ctx, is, fs);
  tensor.region_grad = runtime->create_logical_region(ctx, is, fs);
  // Step 2: create partitions
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, part_is);
  Transform<NDIM, NDIM> transform;
  for (int i = 0; i < NDIM; i++) {
    int nparts = part_rect.hi[i] - part_rect.lo[i] + 1;
    points[i] = (rect.hi[i] - rect.lo[i] + nparts) / nparts - 1;
  }
  Point<NDIM> ext_lo = lo;
  Point<NDIM> ext_hi(points);
  Rect<NDIM> extent(ext_lo, ext_hi);
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
  tensor.part_grad = runtime->get_logical_partition(ctx, tensor.region_grad, ip);
  tensor.numDim = NDIM;
  for (int i = 0; i < NDIM; i++) {
    tensor.adim[i] = rect.hi[i] - rect.lo[i] + 1;
    tensor.pdim[i] = extent.hi[i] - extent.lo[i] + 1;
  }
  return tensor;
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
    pc.dim[i] = domain.hi()[i] - domain.lo()[i];
  }
  return get_or_create_task_is(pc);
}

IndexSpace FFModel::get_or_create_task_is(const std::string& pcname)
{
  assert(config.strategies.find(pcname) != config.strategies.end());
  ParallelConfig pc = config.strategies[pcname];
  return get_or_create_task_is(pc);
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
  for (int p = parameters.size() - 1; p >= 0; p --) {
    optimizer->update(&parameters[p]);
  }
}

void FFModel::zero_gradients(void)
{
  ArgumentMap arg_map;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  for (size_t p = 0; p < parameters.size(); p++) {
    Domain domain = runtime->get_index_partition_color_space(
        ctx, parameters[p].part_grad.get_index_partition());
    IndexSpace task_is = get_or_create_task_is(domain);
    IndexLauncher launcher(ZERO_GRAD_TASK_ID, task_is,
                           TaskArgument(NULL, 0), arg_map);
    launcher.add_region_requirement(
        RegionRequirement(parameters[p].part_grad, 0/*projection*/,
                          WRITE_ONLY, EXCLUSIVE, parameters[p].region_grad));
    launcher.add_field(0, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

void Op::prefetch(const FFModel& ff)
{
  // TODO: perform prefetch for performance imporvement
}

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

// ========================================================
// class FFConfig
// ========================================================

// Default Config Parameters
struct DefaultConfig {
  const static int epochs = 10;
  const static int batchSize = 64;
  const static int inputHeight = 224;
  const static int inputWidth = 224;
  const static bool profiling = false;
  constexpr static float learningRate = 0.01f;
  constexpr static float weightDecay = 0.0001f;
  const static size_t workSpaceSize = (size_t)2 * 1024 * 1024 * 1024; // 2GB
  const static int numNodes = 1;
  const static int workersPerNode = 0;
  const static int loadersPerNode = 4;
};

FFConfig::FFConfig()
{
  epochs = DefaultConfig::epochs;
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
    if (!strcmp(argv[i], "-ll:cpu"))
    {
      loadersPerNode = atoi(argv[++i]);
      continue;
    }
  }
}
