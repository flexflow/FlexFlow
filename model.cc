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

FFModel::FFModel(FFConfig& config)
{
  Runtime *runtime = config.lg_hlr;
  Context ctx = config.lg_ctx;
  // Create field space
  config.field_space = runtime->create_field_space(ctx);
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
  // Build logical regions for images
  Rect<3, coord_t> image_rect(Point<3>(0, 0, 0),
    Point<3>(config.inputWidth-1, config.inputHeight-1, 3*config.batchSize-1));
  IndexSpaceT<3> image_is = runtime->create_index_space(ctx, image_rect);
  LogicalRegion image_lr = runtime->create_logical_region(ctx, image_is,
                               config.field_space);
  LogicalRegion image_grad_lr = runtime->create_logical_region(ctx, image_is,
                                    config.field_space);
  Transform<3, 3, coord_t> transform;
  int extentW = config.inputWidth;
  int extentH = config.inputHeight;
  assert(config.batchSize % (config.numNodes * config.workersPerNode) == 0);
  int extentNC = 3 * config.batchSize / (config.numNodes * config.workersPerNode);
  Rect<3, coord_t> extent(Point<3>(0, 0, 0),
                          Point<3>(extentW-1, extentH-1, extentNC-1));
  transform[0][0] = extentW; transform[0][1] = 0; transform[0][2] = 0;
  transform[1][0] = 0; transform[1][1] = extentH; transform[1][2] = 0;
  transform[2][0] = 0; transform[2][1] = 0; transform[2][2] = extentNC;
  IndexPartition image_ip =
    runtime->create_partition_by_restriction(ctx, image_is, part_is, transform, extent);
  LogicalPartition image_lp = runtime->get_logical_partition(ctx, iamge_lr, image_ip);
  LogicalPartition image_grad_lp =
    runtime->get_logical_partition(ctx, image_grad_lr, image_ip);
  inputImage.numDim = 4;
  inputImage.adim[0] = config.inputWidth;
  inputImage.adim[1] = config.inputHeight;
  inputImage.adim[2] = 3;
  inputImage.adim[3] = config.batchSize;
  inputImage.pdim[0] = extentW;
  inputImage.pdim[1] = extentH;
  inputImage.pdim[2] = 3;
  inputImage.pdim[3] = extentNC / 3;
  inputImage.region = image_lr;
  inputImage.region_grad = image_grad_lr;
  inputImage.partition = image_lp;
  inputImage.partition_grad = image_grad_lp;
  // Build locail regions for input raw images
  Rect<2, coord_t> raw_rect(Point<2>(0, 0), Point<2>(
  // Build logical regions for labels
  
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

