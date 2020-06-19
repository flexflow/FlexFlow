/* Copyright 2017 Stanford, NVIDIA
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
#include "ops.h"
#include "cnn_helper.h"

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 0);
  assert(task->arglen == sizeof(size_t));
  size_t workSpaceSize = *(const size_t*) task->args;
  CnnHandle handle;
  handle.workSpaceSize = workSpaceSize;
  printf("workSpaceSize = %zu\n", workSpaceSize);
#ifndef DISABLE_COMPUTATION
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
#endif
  checkCUDA(cudaMalloc(&handle.workSpace, workSpaceSize));
  return handle;
}

Op::Op(Tensor input)
: numLocals(0)
{
  inputs[0] = input;
}

Op::Op(int n, Tensor *_inputs)
: numLocals(0)
{
  for (int i = 0; i < n; i++) {
    inputs[i] = _inputs[i];
  }
}

void Op::dummy_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{}

void Op::prefetch(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  if (numLocals == 0)
    return;
  //FIXME: this is a hack, fix me later
  if (numLocals == 3) {
    // We must be an Linear operation
    Rect<2> rect = runtime->get_index_space_domain(ctx, model.fc_part_is);
    IndexLauncher launcher(DUMMY_TASK_ID, model.fc_part_is,
                           TaskArgument(NULL, 0), argmap);
    launcher.add_region_requirement(
        RegionRequirement(locals[1].partition, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, locals[1].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(locals[2].partition, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, locals[2].region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  } else {
    assert(numLocals == 2);
    Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
    IndexLauncher launcher(DUMMY_TASK_ID, model.part_is,
                            TaskArgument(NULL, 0), argmap);
    launcher.add_region_requirement(
        RegionRequirement(locals[0].region, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, locals[0].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(locals[1].region, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, locals[1].region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
}

CnnModel::CnnModel(int num_images, int height, int width,
                   int image_par, int height_par, int width_par,
                   int fc_par_n, int fc_par_c, bool profiling,
                   float learning_rate,
                   int num_loaders_per_node, int num_nodes,
                   Context ctx, Runtime* runtime)
{
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  config.field_space = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, config.field_space);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  //config.num_par_w = width_par;
  //config.num_par_h = height_par;
  //config.num_par_n = image_par;
  //config.num_workers = width_par * height_par * image_par;
  //config.fc_num_par_c = fc_par_c;
  //config.fc_num_par_n = fc_par_n;
  config.sm_num_par = fc_par_c * fc_par_n;
  config.profiling = profiling;
  config.learning_rate = learning_rate;
  config.num_loaders = num_loaders_per_node;
  config.num_nodes = num_nodes;
  Rect<3, coord_t> part_bounds(Point<3>(0, 0, 0), Point<3>(width_par-1, height_par-1, image_par-1));
  part_is = runtime->create_index_space(ctx, part_bounds);
  Rect<2, coord_t> fc_part_bounds(Point<2>(0, 0), Point<2>(fc_par_c-1, fc_par_n-1));
  fc_part_is = runtime->create_index_space(ctx, fc_part_bounds);
  Rect<1, coord_t> sm_part_bounds(Point<1>(0), Point<1>(config.sm_num_par-1));
  sm_part_is = runtime->create_index_space(ctx, sm_part_bounds);
  Rect<1, coord_t> load_part_bounds(Point<1>(0), Point<1>(config.num_loaders-1));
  load_part_is = runtime->create_index_space(ctx, load_part_bounds);

  // input_images
  Rect<3, coord_t> image_rect(Point<3>(0, 0, 0), Point<3>(width-1, height-1, num_images*3-1));
  IndexSpaceT<3> image_is = runtime->create_index_space(ctx, image_rect);
  LogicalRegion image_lr = runtime->create_logical_region(ctx, image_is, config.field_space);
  LogicalRegion image_grad_lr = runtime->create_logical_region(ctx, image_is, config.field_space);
  Transform<3, 3, coord_t> transform;
  int extent_w = width / width_par;
  int extent_h = height / height_par;
  int extent_nc = 3 * num_images / image_par;
  Rect<3, coord_t> extent(Point<3>(0, 0, 0), Point<3>(extent_w-1, extent_h-1, extent_nc-1));
  transform[0][0] = extent_w; transform[0][1] = 0; transform[0][2] = 0;
  transform[1][0] = 0; transform[1][1] = extent_h; transform[1][2] = 0;
  transform[2][0] = 0; transform[2][1] = 0; transform[2][2] = extent_nc;
  IndexPartition image_ip = 
    runtime->create_partition_by_restriction(ctx, image_is, part_is, transform, extent);
  LogicalPartition image_lp = runtime->get_logical_partition(ctx, image_lr, image_ip);
  LogicalPartition image_grad_lp =
    runtime->get_logical_partition(ctx, image_grad_lr, image_ip);
  input_image.numDim = 4;
  input_image.adim[0] = width;
  input_image.adim[1] = height;
  input_image.adim[2] = 3;
  input_image.adim[3] = num_images;
  input_image.pdim[0] = extent_w;
  input_image.pdim[1] = extent_h;
  input_image.pdim[2] = 3;
  input_image.pdim[3] = extent_nc / 3;
  input_image.region = image_lr;
  input_image.region_grad = image_grad_lr;
  input_image.partition = image_lp;
  input_image.partition_grad = image_grad_lp;

  // rgb_images (has same index space as input_images
  rgb_lr = runtime->create_logical_region(ctx, image_is, config.field_space);
  rgb_image_lp = runtime->get_logical_partition(ctx, rgb_lr, image_ip);
  // Create a partition based on num_loaders and num_nodes
  assert(num_images * 3 % (config.num_loaders * config.num_nodes) == 0);
  int extent_images = num_images * 3 / (config.num_loaders * config.num_nodes);
  Transform<3, 1, coord_t> trans;
  trans[0][0] = 0; trans[1][0] = 0; trans[2][0] = extent_images;
  Rect<3, coord_t> ext(Point<3>(0, 0, 0), Point<3>(width-1, height-1, extent_images-1));
  IndexPartition rgb_ip =
    runtime->create_partition_by_restriction(ctx, image_is, load_part_is, trans, ext);
  rgb_load_lp = runtime->get_logical_partition(ctx, rgb_lr, rgb_ip);

  // input_label
  Rect<1, coord_t> label_rect(Point<1>(0), Point<1>(num_images-1));
  IndexSpaceT<1> label_is = runtime->create_index_space(ctx, label_rect);
  LogicalRegion label_lr = runtime->create_logical_region(ctx, label_is, config.field_space);
  Transform<1, 1, coord_t> label_trans;
  int extent_n = (num_images + config.sm_num_par - 1) / config.sm_num_par;
  Rect<1, coord_t> label_extent(Point<1>(0), Point<1>(extent_n-1));
  label_trans[0][0] = extent_n;
  IndexPartition label_ip = runtime->create_partition_by_restriction(
                 ctx, label_is, sm_part_is, label_trans, label_extent);
  LogicalPartition label_lp = runtime->get_logical_partition(ctx, label_lr, label_ip);
  input_label.numDim = 1;
  input_label.adim[0] = num_images;
  input_label.pdim[0] = extent_n;
  input_label.region = label_lr;
  input_label.partition = label_lp;

  // Build DataLoader
  dataLoader = new DataLoader("list.txt");
};

__inline__
int calc_offset(int c, int y, int x, int yscale, int xscale)
{
  return (c * yscale * xscale + y * xscale + x);
}

// Note: the layout is CHW in both buffer and image
void bilinear_interpolation(unsigned char* buffer, float *image,
                            int height, int width,
                            int input_height, int input_width,
                            float height_scale, float width_scale)
{
  //printf("h_in(%d) w_in(%d) h_out(%d) w_out(%d) h_scale(%.2lf) w_scale(%.2lf)\n",
  //       input_height, input_width, height, width, height_scale, width_scale);
  //float mean[3] = {0.485, 0.456, 0.406};
  //float variance[3] = {0.229, 0.224, 0.225};
  for (int y = 0; y < height; y++) {
    float input_y = y * height_scale;
    int y0 = static_cast<int>(std::floor(input_y));
    int y1 = std::min(y0 + 1, input_height - 1);
    for (int x = 0; x < width; x++) {
      float input_x = x * width_scale;
      int x0 = static_cast<int>(input_x);
      int x1 = std::min(x0 + 1, input_width - 1);

      // Run kernel on the 4 corners of the bilinear resize algorithm
      float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
      for (int c = 0; c < 3; c++) {
        int input_offset = calc_offset(c, y0, x0, input_height, input_width);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = scale * static_cast<float>(buffer[input_offset]);
      }

      scale = (1 - (input_y - y0)) * (input_x - x0);
      for (int c = 0; c < 3; c++) {
        int input_offset = calc_offset(c, y0, x1, input_height, input_width);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] += scale * static_cast<float>(buffer[input_offset]);
      }

      scale = (input_y - y0) * (1 - (input_x - x0));
      for (int c = 0; c < 3; c++) {
        int input_offset = calc_offset(c, y1, x0, input_height, input_width);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] += scale * static_cast<float>(buffer[input_offset]);
      }

      scale = (input_y - y0) * (input_x - x0);
      for (int c = 0; c < 3; c++) {
        int input_offset = calc_offset(c, y1, x1, input_height, input_width);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] += scale * static_cast<float>(buffer[input_offset]);
      }

      //image[offset] = (image[offset] - mean[c]) / variance[c];
    }
  }
}

// Note: the layout is CHW in both buffer and image
void nearest_neighbor(unsigned char* buffer, unsigned char *image,
                      int height, int width,
                      int input_height, int input_width,
                      float height_scale, float width_scale)
{
  //const float mean[3] = {0.485, 0.456, 0.406};
  //const float variance[3] = {0.229, 0.224, 0.225};
  for (int y = 0; y < height; y++) {
    int y0 = std::min(static_cast<int>(roundf(y * height_scale)), input_height - 1);
    for (int x = 0; x < width; x++) {
      int x0 = std::min(static_cast<int>(roundf(x * width_scale)), input_width - 1);
      for (int c = 0; c < 3; c++) {
        int input_offset = calc_offset(c, y0, x0, input_height, input_width);
        int offset = calc_offset(c, y, x, height, width);
        image[offset] = buffer[input_offset];
        //image[offset] = (static_cast<float>(buffer[input_offset]) / 256 - mean[c]) / variance[c];
      }
    }
  }
}

/*
  regions[0]: image (unsigned char)
  regions[1]: label (int)
*/

void CnnModel::load_images_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  long long start_time = Realm::Clock::current_time_in_microseconds();
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const AccessorWO<unsigned char, 3> acc_image(regions[0], FID_DATA);
  Rect<3> rect_image;
  unsigned char *buffer = (unsigned char*) malloc(2000 * 2000 * 3);
  rect_image = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_image.accessor.is_dense_arbitrary(rect_image));
  unsigned char *image_ptr = acc_image.ptr(rect_image.lo);
  const DataLoadMeta* meta = (DataLoadMeta*) task->local_args;
  int height = rect_image.hi[0] - rect_image.lo[0] + 1;
  int width = rect_image.hi[1] - rect_image.lo[1] + 1;
  int numImages = (rect_image.hi[2] - rect_image.lo[2] + 1) / 3;
  assert((rect_image.hi[2] - rect_image.lo[2] + 1) % 3 == 0);
  for (int fileIdx = 0; fileIdx < meta->cnt; fileIdx ++) {
    //printf("fileIdx = %d filename = %s, start = %d, end = %d\n", fileIdx, meta->datasets[fileIdx].filename, meta->datasets[fileIdx].start, meta->datasets[fileIdx].end);
    hid_t fileId = H5Fopen(meta->datasets[fileIdx].filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    //hid_t fileId = meta->datasets[fileIdx].fid;
    //printf("fileId = %d\n", fileId);
    char name[100];
    for (int i = meta->datasets[fileIdx].start; i <= meta->datasets[fileIdx].end; i++) {
      H5Gget_objname_by_idx(fileId, i, name, 100);
      hid_t datasetId = H5Dopen2(fileId, name, H5P_DEFAULT);
      hid_t dataspaceId = H5Dget_space(datasetId);
      hsize_t dims[3];
      H5Sget_simple_extent_dims(dataspaceId, dims, NULL);
      H5Dread(datasetId, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
      H5Sclose(dataspaceId);
      H5Dclose(datasetId);
      hsize_t input_height = dims[1];
      hsize_t input_width = dims[2];
      //char interlace[100];
      //hsize_t input_height, input_width, input_planes;
      //hssize_t input_npals;
      //H5IMget_image_info(fileId, name, &input_width, &input_height, &input_planes,
      //                   interlace, &input_npals);
      //printf("h = %zu, w = %zu, planes = %zu, npals = %zu\n", input_height, input_width, input_planes, input_npals);
      //H5IMread_image(fileId, name, buffer);
      float height_scale = static_cast<float>(input_height) / height;
      float width_scale = static_cast<float>(input_width) / width;
      nearest_neighbor(buffer, image_ptr, height, width,
                       input_height, input_width, height_scale, width_scale);
      //bilinear_interpolation(buffer, image_ptr, height, width,
      //                       input_height, input_width, height_scale, width_scale);
      image_ptr += 3 * height * width;
    }
    H5Fclose(fileId);
  }
  long long end_time = Realm::Clock::current_time_in_microseconds();
  printf("exe time = %lld\n", end_time - start_time);
  free(buffer);
}

__global__
void apply_normalize(float *tensor_ptr, const unsigned char *rgb_ptr,
                     size_t size, size_t hxw)
{
  const float mean[3] = {0.485, 0.456, 0.406};
  const float var[3] = {0.229, 0.224, 0.225};

  CUDA_KERNEL_LOOP(i, size)
  {
    // decide the color of the current position by assuming NCHW layout
    int c = (i / hxw) % 3;
    tensor_ptr[i] = (static_cast<float>(rgb_ptr[i]) / 256 - mean[c]) / var[c];
  }
}

/*
  regions[0](O): input_images
  regions[1](I): input_rgb
*/
__host__
void CnnModel::normalize_images_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const AccessorWO<float, 3> acc_tensor(regions[0], FID_DATA);
  const AccessorRO<unsigned char, 3> acc_rgb(regions[1], FID_DATA);
  Rect<3> rect_tensor, rect_rgb;
  rect_tensor = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_rgb = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_tensor.accessor.is_dense_arbitrary(rect_tensor));
  assert(acc_rgb.accessor.is_dense_arbitrary(rect_rgb));
  assert(rect_tensor == rect_rgb);
  size_t w = rect_tensor.hi[0] - rect_tensor.lo[0] + 1;
  size_t h = rect_tensor.hi[1] - rect_tensor.lo[1] + 1;
  float *tensor_ptr = acc_tensor.ptr(rect_tensor.lo);
  const unsigned char *rgb_ptr = acc_rgb.ptr(rect_rgb.lo);
  apply_normalize<<<GET_BLOCKS(rect_tensor.volume()), CUDA_NUM_THREADS>>>(
      tensor_ptr, rgb_ptr, rect_tensor.volume(), h * w);
}

void CnnModel::load_images()
{
  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;

  Rect<1> rect = runtime->get_index_space_domain(ctx, load_part_is);
  int total_loaders = config.num_loaders * config.num_nodes;
  assert(input_image.adim[3] % total_loaders == 0);
  int image_per_loader = input_image.adim[3] / total_loaders;
  for (PointInRectIterator<1> it(rect); it(); it++) {
    DataLoadMeta meta;
    dataLoader->get_images(image_per_loader, meta);
    argmap.set_point(*it, TaskArgument(&meta, sizeof(meta)));
  }

  // Load the rgb images
  IndexLauncher launcher(LOAD_IMAGES_TASK_ID, load_part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(rgb_load_lp, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, rgb_lr));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);

  // Conver to float input tensor
  ArgumentMap argmap_dummy;
  IndexLauncher launcher2(NORMALIZE_IMAGES_TASK_ID, part_is,
                         TaskArgument(NULL, 0), argmap_dummy);
  launcher2.add_region_requirement(
      RegionRequirement(input_image.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, input_image.region));
  launcher2.add_field(0, FID_DATA);
  launcher2.add_region_requirement(
      RegionRequirement(rgb_image_lp, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, rgb_lr));
  launcher2.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher2);
}

void CnnModel::prefetch()
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->prefetch(*this);
}

void CnnModel::forward()
{
  for (size_t i = 0; i < layers.size(); i++) {
    layers[i]->forward(*this);
  }
}

void CnnModel::backward()
{
  for (int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->backward(*this);
  }
}

void CnnModel::update()
{
  for (int i = layers.size() - 1; i >= 0; i--) {
    layers[i]->update(*this);
  }
}

__global__
void init_image_kernel(float* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1.0f;
  }
}

__global__
void init_label_kernel(int* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1;
  }
}

void CnnModel::init_images_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  const int BLKSIZE = 512;
  const AccessorWO<float, 3> acc_image(regions[0], FID_DATA);
  Rect<3> rect_image;
  rect_image = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_image.accessor.is_dense_arbitrary(rect_image));
  float *image_ptr = acc_image.ptr(rect_image.lo);
  int num_blocks = (rect_image.volume() + BLKSIZE - 1) / BLKSIZE;
  init_image_kernel<<<num_blocks, BLKSIZE>>>(image_ptr, rect_image.volume());
#endif
}

void CnnModel::init_images()
{
  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  IndexLauncher launcher(IMAGE_INIT_TASK_ID, part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_image.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, input_image.region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void CnnModel::init_labels_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
  const int BLKSIZE = 512;
  const AccessorWO<int, 1> acc_label(regions[0], FID_DATA);
  Rect<1> rect_label;
  rect_label = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  int *label_ptr = acc_label.ptr(rect_label.lo);
  int num_blocks = (rect_label.volume() + BLKSIZE - 1) / BLKSIZE;
  init_label_kernel<<<num_blocks, BLKSIZE>>>(label_ptr, rect_label.volume());
}

void CnnModel::init_labels()
{
  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  IndexLauncher launcher(LABEL_INIT_TASK_ID, sm_part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_label.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, input_label.region));
  launcher.add_field(0, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  //fm.wait_all_results();
}

Tensor CnnModel::add_flat_layer(Tensor input)
{
  assert(input.numDim == 4);
  Flat *flat = new Flat(config, input, part_is, fc_part_is);
  layers.push_back(flat);
  return flat->output;
}

Flat::Flat(CnnConfig config, Tensor input,
           IndexSpaceT<3> part_is_3d,
           IndexSpaceT<2> part_is_2d)
: Op(input)
{
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;
  Rect<2> part_rect_2d = runtime->get_index_space_domain(ctx, part_is_2d);
  int fc_num_par_c = part_rect_2d.hi[0] - part_rect_2d.lo[0] + 1;
  int fc_num_par_n = part_rect_2d.hi[1] - part_rect_2d.lo[1] + 1;
 
  FieldSpace fs = config.field_space;
  
  int output_c = input.adim[0] * input.adim[1] * input.adim[2];
  int output_n = input.adim[3];
  Rect<2, coord_t> output_rect(Point<2>(0, 0), Point<2>(output_c-1, output_n-1));
  IndexSpaceT<2> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr =
    runtime->create_logical_region(ctx, output_is, fs);
  Transform<2, 2, coord_t> transform;
  //int extent_c = input.pdim[0] * input.pdim[1] * input.pdim[2];
  //int extent_n = input.pdim[3];
  // We assume equal partition for load balancing
  assert(output_c % fc_num_par_c == 0);
  assert(output_n % fc_num_par_n == 0);
  int extent_c = output_c / fc_num_par_c;
  int extent_n = output_n / fc_num_par_n;
  Rect<2, coord_t> extent(Point<2>(0, 0), Point<2>(extent_c-1,extent_n-1));
  transform[0][0] = extent_c; transform[0][1] = 0;
  transform[1][0] = 0; transform[1][1] = extent_n;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is_2d, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, output_ip));
  assert(runtime->is_index_partition_complete(ctx, output_ip));
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp =
    runtime->get_logical_partition(ctx, output_grad_lr, output_ip);
  output.numDim = 2;
  output.adim[0] = output_c;
  output.adim[1] = output_n;
  output.pdim[0] = extent_c;
  output.pdim[1] = extent_n;
  output.region = output_lr;
  output.region_grad = output_grad_lr;
  output.partition = output_lp;
  output.partition_grad = output_grad_lp;
  printf("Create flat layer: input(N=%d C=%d H=%d W=%d) -> output(N=%d C=%d)\n",
         input.adim[3], input.adim[2], input.adim[1], input.adim[0], output.adim[1], output.adim[0]);
 
  FieldSpace proj_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, proj_fs);
    allocator.allocate_field(sizeof(Rect<2>), FID_DATA);
  }
  LogicalRegion proj_lr = runtime->create_logical_region(ctx, part_is_3d, proj_fs);
  InlineLauncher launcher(RegionRequirement(proj_lr, WRITE_DISCARD, EXCLUSIVE, proj_lr)
                                           .add_field(FID_DATA));
  PhysicalRegion proj_pr = runtime->map_region(ctx, launcher);
  proj_pr.wait_until_valid();
  coord_t subtotal = 0;
  {
    const FieldAccessor<WRITE_DISCARD, Rect<2>, 3, coord_t,
              Realm::AffineAccessor<Rect<2>, 3, coord_t> > ra(proj_pr, FID_DATA);
    Rect<3> rect = runtime->get_index_space_domain(ctx, part_is_3d);
    for(PointInRectIterator<3> pir(rect); pir(); ++pir) {
      IndexSpace subspace = runtime->get_index_subspace(input.partition.get_index_partition(), *pir);
      Rect<3> subrect = runtime->get_index_space_domain(ctx, subspace);
      // Currently we assume the size of each subregion is divisible by output_n (i.e., batch size)
      assert(subrect.volume() % output_n == 0);
      coord_t subsize = subrect.volume() / output_n;
      ra[*pir] = Rect<2>(Point<2>(subtotal, 0), Point<2>(subtotal + subsize - 1, output_n - 1));
      subtotal += subsize;
    }
  }
  runtime->unmap_region(ctx, proj_pr);
  Transform<3, 3, coord_t> proj_trans;
  proj_trans[0][0] = 1; proj_trans[0][1] = 0; proj_trans[0][2] = 0;
  proj_trans[1][0] = 0; proj_trans[1][1] = 1; proj_trans[1][2] = 0;
  proj_trans[2][0] = 0; proj_trans[2][1] = 0; proj_trans[2][2] = 1;
  Rect<3, coord_t> proj_extent(Point<3>(0, 0, 0), Point<3>(0, 0, 0));
  IndexPartition proj_ip =
    runtime->create_partition_by_restriction(ctx, part_is_3d, part_is_3d, proj_trans, proj_extent);
  LogicalPartition proj_lp = runtime->get_logical_partition(ctx, proj_lr, proj_ip);
  IndexPartition flat_ip =
    runtime->create_partition_by_image_range(ctx, output_is,
                         proj_lp, proj_lr, FID_DATA, part_is_3d);
  assert(runtime->is_index_partition_disjoint(ctx, flat_ip));
  assert(runtime->is_index_partition_complete(ctx, flat_ip));
  flat_lp = runtime->get_logical_partition(ctx, output_lr, flat_ip);
  flat_grad_lp = runtime->get_logical_partition(ctx, output_grad_lr, flat_ip);
  return;
/*
  Transform<2, 3, coord_t> flat_trans;
  flat_trans[0][0] = input.pdim[0] * input.pdim[1] * input.adim[2];
  flat_trans[0][1] = input.adim[0] * input.pdim[1] * input.adim[2];
  flat_trans[0][2] = 0;
  flat_trans[1][0] = 0;
  flat_trans[1][1] = 0;
  flat_trans[1][2] = input.pdim[3];
  IndexPartition flat_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is_3d, flat_trans, extent);
  flat_lp = runtime->get_logical_partition(ctx, output_lr, flat_ip);
*/
}

OpMeta* Flat::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  CnnHandle handle = *((const CnnHandle*) task->local_args);
  FlatMeta* m = new FlatMeta(handle);
  return m;
}

void Flat::init(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    CnnHandle handle = model.cnn_handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(CnnHandle)));
  }

  IndexLauncher init_launcher(FLAT_INIT_TASK_ID, model.part_is,
                              TaskArgument(this, sizeof(Flat)), argmap);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/  
void Flat::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const AccessorRO<float, 3> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  Rect<3> rect_input;
  Rect<2> rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(rect_input.volume() == rect_output.volume());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);

  checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
                            rect_input.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
#endif
}

void Flat::forward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(FLAT_FWD_TASK_ID, model.part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].partition, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(flat_lp /*3D->2D partitions*/, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](O) : input_grad
  regions[1](I) : output_grad
*/
void Flat::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const AccessorWO<float, 3> acc_input_grad(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_output_grad(regions[1], FID_DATA);
  Rect<3> rect_input_grad;
  Rect<2> rect_output_grad;
  rect_input_grad =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(rect_input_grad.volume() == rect_output_grad.volume());
  assert(acc_input_grad.accessor.is_dense_arbitrary(rect_input_grad));
  assert(acc_output_grad.accessor.is_dense_arbitrary(rect_output_grad));
  float *input_grad_ptr = acc_input_grad.ptr(rect_input_grad.lo);
  const float *output_grad_ptr = acc_output_grad.ptr(rect_output_grad.lo);

  checkCUDA(cudaMemcpyAsync(input_grad_ptr, output_grad_ptr,
                            rect_input_grad.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
#endif
}

void Flat::backward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(FLAT_BWD_TASK_ID, model.part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].partition_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(flat_grad_lp /*3D->2D partitions*/, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Flat::update(const CnnModel& model)
{
}

DataLoader::DataLoader(std::string filename)
  : fileIdx(0), imageIdx(0)
{
  FILE *file;
  file = fopen(filename.c_str(), "r");
  assert(file != NULL);
  HDFFile hdf;
  while (fgets(hdf.filename, MAX_FILENAME, file) != NULL) {
    hdf.filename[strlen(hdf.filename) - 1] = 0;
    printf("filename = %s\n", hdf.filename);
    hid_t fileId = H5Fopen(hdf.filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    H5Gget_num_objs(fileId, &hdf.numImages);
    hdf.fid = fileId;
    //H5Fclose(fileId);
    datasets.push_back(hdf);
  }
}

void DataLoader::get_images(int numImages, DataLoadMeta &meta)
{
  int idx = 0;
  if (imageIdx == (int)datasets[fileIdx].numImages) {
    imageIdx = 0;
    fileIdx = (fileIdx + 1) % datasets.size();
  }
  memcpy(meta.datasets[0].filename, datasets[fileIdx].filename, MAX_FILENAME);
  meta.datasets[0].fid = datasets[fileIdx].fid;
  meta.datasets[0].start = imageIdx;
  meta.datasets[0].end = imageIdx;
  for (int i = 0; i < numImages; i++) {
    if (imageIdx < (int)datasets[fileIdx].numImages) {
      meta.datasets[idx].end = imageIdx;
      imageIdx ++;
    } else {
      imageIdx = 0;
      fileIdx = (fileIdx + 1) % datasets.size();
      idx++;
      memcpy(meta.datasets[idx].filename, datasets[fileIdx].filename, MAX_FILENAME);
      meta.datasets[idx].fid = datasets[fileIdx].fid;
      meta.datasets[idx].start = imageIdx;
      meta.datasets[idx].end = imageIdx;
    }
  }
  meta.cnt = idx + 1;
  printf("meta.cnt = %d\n", meta.cnt);
  for (int i = 0; i < meta.cnt; i++)
    printf("fn = %s, start = %d, end = %d\n", meta.datasets[i].filename, meta.datasets[i].start, meta.datasets[i].end);
}

