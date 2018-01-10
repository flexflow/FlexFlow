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
{
  inputs[0] = input;
}

Op::Op(int n, Tensor *_inputs)
{
  for (int i = 0; i < n; i++) {
    inputs[i] = _inputs[i];
  }
}

CnnModel::CnnModel(int num_images, int height, int width,
                   int image_par, int height_par, int width_par,
                   int fc_par_n, int fc_par_c, bool profiling,
                   float learning_rate,
                   Context ctx, Runtime* runtime)
{
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  //config.num_par_w = width_par;
  //config.num_par_h = height_par;
  //config.num_par_n = image_par;
  //config.num_workers = width_par * height_par * image_par;
  //config.fc_num_par_c = fc_par_c;
  //config.fc_num_par_n = fc_par_n;
  config.sm_num_par = fc_par_c * fc_par_n;
  config.profiling = profiling;
  config.learning_rate = learning_rate;
  Rect<3, coord_t> part_bounds(Point<3>(0, 0, 0), Point<3>(width_par-1, height_par-1, image_par-1));
  part_is = runtime->create_index_space(ctx, part_bounds);
  Rect<2, coord_t> fc_part_bounds(Point<2>(0, 0), Point<2>(fc_par_c-1, fc_par_n-1));
  fc_part_is = runtime->create_index_space(ctx, fc_part_bounds);
  Rect<1, coord_t> sm_part_bounds(Point<1>(0), Point<1>(config.sm_num_par-1));
  sm_part_is = runtime->create_index_space(ctx, sm_part_bounds);

  // input_images
  Rect<3, coord_t> image_rect(Point<3>(0, 0, 0), Point<3>(width-1, height-1, num_images*3-1));
  IndexSpaceT<3> image_is = runtime->create_index_space(ctx, image_rect);
  FieldSpace image_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, image_fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  LogicalRegion image_lr = runtime->create_logical_region(ctx, image_is, image_fs);
  LogicalRegion image_grad_lr = runtime->create_logical_region(ctx, image_is, image_fs);
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

  // input_label
  Rect<1, coord_t> label_rect(Point<1>(0), Point<1>(num_images-1));
  IndexSpaceT<1> label_is = runtime->create_index_space(ctx, label_rect);
  FieldSpace label_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, label_fs);
    allocator.allocate_field(sizeof(int), FID_DATA);
  }
  LogicalRegion label_lr = runtime->create_logical_region(ctx, label_is, label_fs);
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
  printf("h_in(%d) w_in(%d) h_out(%d) w_out(%d) h_scale(%.2lf) w_scale(%.2lf)\n",
         input_height, input_width, height, width, height_scale, width_scale);
  int offset = 0;
  float mean[3] = {0.485, 0.456, 0.406};
  float variance[3] = {0.229, 0.224, 0.225};
  for (int c = 0; c < 3; c++)
    for (int y = 0; y < height; y++) {
      float input_y = y * height_scale;
      int y0 = static_cast<int>(std::floor(input_y));
      int y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < width; x++) {
        float input_x = x * width_scale;
        int x0 = static_cast<int>(input_x);
        int x1 = std::min(x0 + 1, input_width - 1);

        // Run kernel on the 4 corners of the bilinear resize algorithm
        image[offset] = 0;
        int input_offset = calc_offset(c, y0, x0, input_height, input_width);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        image[offset] += scale * static_cast<float>(buffer[input_offset]) / 256;

        input_offset = calc_offset(c, y0, x1, input_height, input_width);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        image[offset] += scale * static_cast<float>(buffer[input_offset]) / 256;

        input_offset = calc_offset(c, y1, x0, input_height, input_width);
        scale = (input_y - y0) * (1 - (input_x - x0));
        image[offset] += scale * static_cast<float>(buffer[input_offset]) / 256;

        input_offset = calc_offset(c, y1, x1, input_height, input_width);
        scale = (input_y - y0) * (input_x - x0);
        image[offset] += scale * static_cast<float>(buffer[input_offset]) / 256;

        image[offset] = (image[offset] - mean[c]) / variance[c];
        offset ++;
      }
    }
}

void CnnModel::load_images_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const AccessorWO<float, 3> acc_image(regions[0], FID_DATA);
  Rect<3> rect_image;
  unsigned char buffer[2000 * 2000 * 3];
  rect_image = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_image.accessor.is_dense_arbitrary(rect_image));
  float *image_ptr = acc_image.ptr(rect_image.lo);
  const DataLoadMeta* meta = (DataLoadMeta*) task->local_args;
  int height = rect_image.hi[0] - rect_image.lo[0] + 1;
  int width = rect_image.hi[1] - rect_image.lo[1] + 1;
  int numImages = (rect_image.hi[2] - rect_image.lo[2] + 1) / 3;
  assert((rect_image.hi[2] - rect_image.lo[2] + 1) % 3 == 0);
  for (int fileIdx = 0; fileIdx < meta->cnt; fileIdx ++) {
    hid_t fileId = H5Fopen(meta->datasets[fileIdx].filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    char name[100];
    for (int i = meta->datasets[fileIdx].start; i <= meta->datasets[fileIdx].end; i++) {
      H5Gget_objname_by_idx(fileId, i, name, 100);
      hsize_t input_height, input_width, input_planes;
      hssize_t input_npals;
      char interlace[100];
      H5IMget_image_info(fileId, name, &input_width, &input_height, &input_planes,
                         interlace, &input_npals);
      H5IMread_image(fileId, name, buffer);
      float height_scale = static_cast<float>(input_height) / height;
      float width_scale = static_cast<float>(input_width) / width;
      bilinear_interpolation(buffer, image_ptr, height, width,
                             input_height, input_width, height_scale, width_scale);
      image_ptr += 3 * height * width;
    }
    H5Fclose(fileId);
  }
}

void CnnModel::load_images()
{
  //TODO: use a new partition to load images, for now we reuse image_part
  // by assuming par_h == par_w == 1
  int image_per_task = input_image.pdim[3];
  assert(input_image.pdim[0] == input_image.adim[0]);
  assert(input_image.pdim[1] == input_image.adim[1]);
  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;

  Rect<3> rect = runtime->get_index_space_domain(ctx, part_is);
  for (PointInRectIterator<3> it(rect); it(); it++) {
    DataLoadMeta meta;
    dataLoader->get_images(image_per_task, meta);
    argmap.set_point(*it, TaskArgument(&meta, sizeof(meta)));
  }
  IndexLauncher launcher(LOAD_IMAGES_TASK_ID, part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_image.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, input_image.region));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
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
 
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  
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
    H5Fclose(fileId);
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
      memcpy(meta.datasets[0].filename, datasets[fileIdx].filename, MAX_FILENAME);
      meta.datasets[idx].start = imageIdx;
      meta.datasets[idx].end = imageIdx;
    }
  }
  meta.cnt = idx + 1;
  printf("meta.cnt = %d\n", meta.cnt);
  for (int i = 0; i < meta.cnt; i++)
    printf("fn = %s, start = %d, end = %d\n", meta.datasets[i].filename, meta.datasets[i].start, meta.datasets[i].end);
}

