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

