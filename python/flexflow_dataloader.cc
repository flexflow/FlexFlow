/* Copyright 2020 Stanford, Los Alamos National Laboratory
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

#include <sstream>
#include <fstream>
#include <string>
#include "flexflow_dataloader.h"


ImgDataLoader::ImgDataLoader()
{}

void ImgDataLoader::reset()
{
  next_index = 0;
}

ImgDataLoader4D::ImgDataLoader4D(FFModel& ff, Tensor input, Tensor label, 
                                 Tensor full_input_, Tensor full_label_, 
                                 int num_samples_)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = num_samples_;
  // Create full input
  {
    batch_input = input;
    const int dims[] = {num_samples, input.adim[2], input.adim[1], input.adim[0]};
    full_input = ff.create_tensor<4>(dims, "", DT_FLOAT);
  }
  // Create full label
  {
    batch_label = label;
    const int dims[] = {num_samples, label.adim[0]};
    full_label = ff.create_tensor<2>(dims, "", DT_INT32);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_2,
      TaskArgument(NULL, 0));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input.region, WRITE_ONLY,
                        EXCLUSIVE, full_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region, WRITE_ONLY,
                        EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: full_input_
  launcher.add_region_requirement(
      RegionRequirement(full_input_.region, READ_ONLY,
                        EXCLUSIVE, full_input_.region));
  launcher.add_field(2, FID_DATA);
  // regions[3]: full_label_
  launcher.add_region_requirement(
      RegionRequirement(full_label_.region, READ_ONLY,
                        EXCLUSIVE, full_label_.region));
  launcher.add_field(3, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
  reset();
  next_batch(ff);
}

ImgDataLoader4D::ImgDataLoader4D(FFModel& ff, 
                                 const NetConfig& alexnet,
                                 Tensor input, Tensor label)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = 0;
  if (alexnet.dataset_path == "") {
    printf("Use random dataset...");
    num_samples = 256 * 10 * ff.config.workersPerNode * ff.config.numNodes;
    printf("Number of random samples = %d\n", num_samples);
  } else {
    printf("Start loading dataset from %s\n", alexnet.dataset_path.c_str());
    size_t filesize = get_file_size(alexnet.dataset_path);
    assert(filesize % 3073 == 0);
    num_samples = filesize / 3073;
  }
  // Create full input
  {
    batch_input = input;
    const int dims[] = {num_samples, input.adim[2], input.adim[1], input.adim[0]};
    full_input = ff.create_tensor<4>(dims, "", DT_FLOAT);
  }
  // Create full label
  {
    batch_label = label;
    const int dims[] = {num_samples, label.adim[0]};
    full_label = ff.create_tensor<2>(dims, "", DT_INT32);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  const NetConfig* ptr = &alexnet;
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
      TaskArgument(&ptr, sizeof(NetConfig*)));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input.region, WRITE_ONLY,
                        EXCLUSIVE, full_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region, WRITE_ONLY,
                        EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  runtime->execute_task(ctx, launcher);
  reset();
  next_batch(ff);
}

void ImgDataLoader4D::load_entire_dataset_from_numpy(const Task *task,
                                                    const std::vector<PhysicalRegion> &regions,
                                                    Context ctx, Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == regions.size());
  const AccessorWO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<int, 2> acc_label(regions[1], FID_DATA);
  const AccessorRO<float, 4> acc_input_(regions[2], FID_DATA);
  const AccessorRO<int, 2> acc_label_(regions[3], FID_DATA);
  Rect<4> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<4> rect_input_ = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_input_.accessor.is_dense_arbitrary(rect_input_));
  Rect<2> rect_label_ = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  assert(acc_label_.accessor.is_dense_arbitrary(rect_label_));
  float* input_ptr = acc_input.ptr(rect_input.lo);
  int* label_ptr = acc_label.ptr(rect_label.lo);
  const float* input_ptr_ = acc_input_.ptr(rect_input_.lo);
  const int* label_ptr_ = acc_label_.ptr(rect_label_.lo);
  printf("Check ptr input %p %lu %lu, label %p %lu %lu\n", input_ptr_, (uintptr_t)input_ptr_, rect_input.volume(), label_ptr_, (uintptr_t)label_ptr_, rect_label.volume());
  int num_samples = rect_label.hi[1] - rect_label.lo[1] + 1;
  assert(rect_input.hi[3] - rect_input.lo[3] + 1 == num_samples);
  assert(rect_label.volume() == rect_label_.volume());
  assert(rect_input.volume() == rect_input_.volume());
  memcpy(input_ptr, input_ptr_, sizeof(float)*rect_input.volume());
  memcpy(label_ptr, label_ptr_, sizeof(int)*rect_label.volume());
  for (int i = 0; i < 32; i++) {
    printf("%f ", input_ptr[i]);
  }
  printf("\n");
}

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}

void nearest_neigh(unsigned char* image,
                   unsigned char* buffer,
                   int height, int width,
                   int orig_height, int orig_width,
                   float height_scale, float width_scale)
{
  for (int y = 0; y < height; y++) {
    int y0 = std::min(static_cast<int>(roundf(y * height_scale)), orig_height - 1);
    for (int x = 0; x < width; x++) {
      int x0 = std::min(static_cast<int>(roundf(x * width_scale)), orig_width - 1);
      for (int c = 0; c < 3; c++) {
        int origOffset = calc_offset(y0, x0, c, orig_width, 3);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = buffer[origOffset];
      }
    }
  }
}

void ImgDataLoader4D::load_entire_dataset(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, Runtime* runtime)
{
  const NetConfig* alexnet = *((NetConfig**)task->args);
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  const AccessorWO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<int, 2> acc_label(regions[1], FID_DATA);
  Rect<4> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float* input_ptr = acc_input.ptr(rect_input.lo);
  int* label_ptr = acc_label.ptr(rect_label.lo);
  int num_samples = rect_label.hi[1] - rect_label.lo[1] + 1;
  assert(rect_input.hi[3] - rect_input.lo[3] + 1 == num_samples);
  if (alexnet->dataset_path.length() == 0) {
    printf("Start generating random input samples\n");
    for (size_t i = 0; i < rect_label.volume(); i++)
      label_ptr[i] = std::rand() % 10;
    return;
  }
  printf("Start loading %d samples from %s\n",
      num_samples, alexnet->dataset_path.c_str());
  int height = rect_input.hi[1] - rect_input.lo[1] + 1;
  int width = rect_input.hi[0] - rect_input.lo[0] + 1;
  int origHeight = 32;
  int origWidth = 32;
  float heightScale = static_cast<float>(origHeight) / height;
  float widthScale = static_cast<float>(origWidth) / width;
  FILE* file = fopen(alexnet->dataset_path.c_str(), "rb");
  unsigned char* buffer = (unsigned char*) malloc(3073);
  unsigned char* image = (unsigned char*) malloc(3 * height * width);
  for (off_t i = 0; i < num_samples; i++) {
    size_t ret = fread(buffer, sizeof(unsigned char), 3073, file);
    assert(ret = 3073);
    if (i == 0) {
      for (int i = 0; i < 32; i++) {
        printf("%f ", static_cast<float>(buffer[i])/255);
      }
      printf("\n");
    }
    if ((i+1) % 1000 == 0) {
      printf("Loaded %d samples\n", i+1);
    }
    label_ptr[i] = buffer[0];
    nearest_neigh(image, buffer + 1, height, width,
                  origHeight, origWidth, heightScale, widthScale);
    off_t input_offset = i * 3 * height * width;
    off_t image_offset = 0;
    for (off_t h = 0; h < 3*height*width; h++)
        input_ptr[input_offset++] = static_cast<float>(image[image_offset++]) / 255;
  }
  printf("Finish loading %d samples from %s\n",
      num_samples, alexnet->dataset_path.c_str());
  fclose(file);
  for (int i = 0; i < 32; i++) {
    printf("%f ", input_ptr[i]);
  }
  printf("\n");
}

void ImgDataLoader4D::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load input
  {
    IndexSpaceT<4> task_is = IndexSpaceT<4>(ff.get_or_create_task_is(4, ""));
    Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<4> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[3] - rect.lo[3] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[3] - rect.lo[3] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_label.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

size_t ImgDataLoader4D::get_file_size(const std::string& filename)
{
  std::streampos begin,end;
  std::ifstream file(filename.c_str(), std::ios::binary);
  begin = file.tellg();
  file.seekg (0, std::ios::end);
  end = file.tellg();
  file.close();
  size_t filesize = end - begin;
  return filesize;
}

ImgDataLoader2D::ImgDataLoader2D(FFModel& ff, Tensor input, Tensor label, 
                                 Tensor full_input_, Tensor full_label_, 
                                 int num_samples_)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = num_samples_;
  // Create full input
  {
    batch_input = input;
    const int dims[] = {num_samples, input.adim[0]};
    full_input = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  // Create full label
  {
    batch_label = label;
    const int dims[] = {num_samples, label.adim[0]};
    full_label = ff.create_tensor<2>(dims, "", DT_INT32);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_3,
      TaskArgument(NULL, 0));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input.region, WRITE_ONLY,
                        EXCLUSIVE, full_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region, WRITE_ONLY,
                        EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: full_input_
  launcher.add_region_requirement(
      RegionRequirement(full_input_.region, READ_ONLY,
                        EXCLUSIVE, full_input_.region));
  launcher.add_field(2, FID_DATA);
  // regions[3]: full_label_
  launcher.add_region_requirement(
      RegionRequirement(full_label_.region, READ_ONLY,
                        EXCLUSIVE, full_label_.region));
  launcher.add_field(3, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
  reset();
  next_batch(ff);
}

void ImgDataLoader2D::load_entire_dataset_from_numpy(const Task *task,
                                                     const std::vector<PhysicalRegion> &regions,
                                                     Context ctx, Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == regions.size());
  const AccessorWO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<int, 2> acc_label(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_input_(regions[2], FID_DATA);
  const AccessorRO<int, 2> acc_label_(regions[3], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<2> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_input_ = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_input_.accessor.is_dense_arbitrary(rect_input_));
  Rect<2> rect_label_ = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  assert(acc_label_.accessor.is_dense_arbitrary(rect_label_));
  float* input_ptr = acc_input.ptr(rect_input.lo);
  int* label_ptr = acc_label.ptr(rect_label.lo);
  const float* input_ptr_ = acc_input_.ptr(rect_input_.lo);
  const int* label_ptr_ = acc_label_.ptr(rect_label_.lo);
  printf("Check ptr input %p %lu %lu, label %p %lu %lu\n", input_ptr_, (uintptr_t)input_ptr_, rect_input.volume(), label_ptr_, (uintptr_t)label_ptr_, rect_label.volume());
  int num_samples = rect_label.hi[1] - rect_label.lo[1] + 1;
  assert(rect_input.hi[1] - rect_input.lo[1] + 1 == num_samples);
  assert(rect_label.volume() == rect_label_.volume());
  assert(rect_input.volume() == rect_input_.volume());
  memcpy(input_ptr, input_ptr_, sizeof(float)*rect_input.volume());
  memcpy(label_ptr, label_ptr_, sizeof(int)*rect_label.volume());
  for (int i = 0; i < 32; i++) {
    printf("%f ", input_ptr[i]);
  }
  printf("\n");
}

void ImgDataLoader2D::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load input
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_3, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load label
  {
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(2, ""));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_label.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

SingleDataLoader::SingleDataLoader(FFModel& ff, Tensor input, Tensor full_input_, int num_samples_, DataType datatype_)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = num_samples_;
  datatype = datatype_;
  // Create full input
  assert(input.numDim == full_input_.numDim);
  if (input.numDim == 4) {
    batch_input = input;
    const int dims[] = {num_samples, input.adim[2], input.adim[1], input.adim[0]};
    full_input = ff.create_tensor<4>(dims, "", datatype);
  } else if(input.numDim == 2) {
    batch_input = input;
    const int dims[] = {num_samples, input.adim[0]};
    full_input = ff.create_tensor<2>(dims, "", datatype);
  } else {
    assert(0);
  }
  int task_id = -1;
  if (input.numDim == 4 && datatype == DT_FLOAT) {
    task_id = CUSTOM_CPU_TASK_ID_4;
  } else if (input.numDim == 2 && datatype == DT_FLOAT) {
    task_id = CUSTOM_CPU_TASK_ID_5;
  } else if (input.numDim == 2 && datatype == DT_INT32) {
    task_id = CUSTOM_CPU_TASK_ID_6;
  } else {
    assert(0);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(task_id,
      TaskArgument(NULL, 0));
  // regions[0]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input.region, WRITE_ONLY,
                        EXCLUSIVE, full_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[2]: full_input_
  launcher.add_region_requirement(
      RegionRequirement(full_input_.region, READ_ONLY,
                        EXCLUSIVE, full_input_.region));
  launcher.add_field(1, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
  reset();
  next_batch(ff);
}

void SingleDataLoader::reset()
{
  next_index = 0;
}

void SingleDataLoader::next_batch(FFModel& ff)
{
  if (full_input.numDim == 4 && datatype == DT_FLOAT) {
    int task_id = CUSTOM_GPU_TASK_ID_4;
    next_batch_xd_launcher<4>(ff, task_id);
  } else if (full_input.numDim == 2 && datatype == DT_FLOAT) {
    int task_id = CUSTOM_GPU_TASK_ID_5;
    next_batch_xd_launcher<2>(ff, task_id);
  } else if (full_input.numDim == 2 && datatype == DT_INT32) {
    int task_id = CUSTOM_GPU_TASK_ID_6;
    next_batch_xd_launcher<2>(ff, task_id);
  } else {
    assert(0);
  }
}

template<int NDIM>
void SingleDataLoader::next_batch_xd_launcher(FFModel& ff, int task_id)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load input
  {
    IndexSpaceT<NDIM> task_is = IndexSpaceT<NDIM>(ff.get_or_create_task_is(NDIM, ""));
    Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<NDIM> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[NDIM-1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[NDIM-1] - rect.lo[NDIM-1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(task_id, task_is,
                           TaskArgument(NULL,0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(
        RegionRequirement(full_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
}

// Task body
template<typename DT, int NDIM>
void SingleDataLoader::load_entire_dataset_from_numpy(const Task *task,
                                                      const std::vector<PhysicalRegion> &regions,
                                                      Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  const AccessorWO<DT, NDIM> acc_input(regions[0], FID_DATA);
  const AccessorRO<DT, NDIM> acc_input_(regions[1], FID_DATA);
  Rect<NDIM> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<NDIM> rect_input_ = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_input_.accessor.is_dense_arbitrary(rect_input_));
  assert(rect_input_.volume() == rect_input.volume());
    
  DT* input_ptr = acc_input.ptr(rect_input.lo);
  const DT* input_ptr_ = acc_input_.ptr(rect_input_.lo);
  printf("Check ptr input_ %p %lu %lu, input %p %lu %lu\n", input_ptr_, (uintptr_t)input_ptr_, rect_input_.volume(), input_ptr, (uintptr_t)input_ptr, rect_input.volume());
  assert(rect_input.volume() == rect_input_.volume());
  memcpy(input_ptr, input_ptr_, sizeof(DT)*rect_input.volume());
  for (int i = 0; i < 32; i++) {
    std::cout<<input_ptr[i]<<" ";
  }
  std::cout<<std::endl;
}

void SingleDataLoader::register_cpu_tasks(void)
{
  // 4D float Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_4, "4D Float Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SingleDataLoader::load_entire_dataset_from_numpy<float, 4>>(
        registrar, "4D Float Load Entire Dataset Task Numpy");
  }
  
  // 2D float Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_5, "2D Float Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SingleDataLoader::load_entire_dataset_from_numpy<float, 2>>(
        registrar, "2D Float Load Entire Dataset Task Numpy");
  }
  
  // 2D int Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_6, "2D Int Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SingleDataLoader::load_entire_dataset_from_numpy<int, 2>>(
        registrar, "2D Int Load Entire Dataset Task Numpy");
  }
}

void SingleDataLoader::register_gpu_tasks(void)
{
  // 4D float load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_4, "4D Float Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SingleDataLoader::load_input_4d<float>>(
        registrar, "4D Float Load Input Task");
  }
  // 2D float load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_5, "2D Float Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SingleDataLoader::load_input_2d<float>>(
        registrar, "2D Float Load Input Task");
  }
  // 2D int load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_6, "2D int Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SingleDataLoader::load_input_2d<int>>(
        registrar, "2D Int Load Input Task");
  }
}

template void SingleDataLoader::next_batch_xd_launcher<2>(FFModel& ff, int task_id);
template void SingleDataLoader::next_batch_xd_launcher<4>(FFModel& ff, int task_id);
template void SingleDataLoader::load_entire_dataset_from_numpy<float, 4>(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime* runtime);
template void SingleDataLoader::load_entire_dataset_from_numpy<float, 2>(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime* runtime);
template void SingleDataLoader::load_entire_dataset_from_numpy<int, 2>(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime* runtime);
