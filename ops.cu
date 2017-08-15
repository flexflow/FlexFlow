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
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
  checkCUDA(cudaMalloc(&handle.workSpace, workSpaceSize));
  return handle;
}

Op::Op(Tensor input)
{
  inputs[0] = input;
}

CnnModel::CnnModel(int num_images, int height, int width,
                   int image_par, int height_par, int width_par,
                   Context ctx, Runtime* runtime)
{
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  config.num_par_w = width_par;
  config.num_par_h = height_par;
  config.num_par_n = image_par;
  config.num_workers = width_par * height_par * image_par;
  Realm::ZRect<3, coord_t> part_bounds(Realm::ZPoint<3>(0, 0, 0),
                             Realm::ZPoint<3>(width_par-1, height_par-1, image_par-1));
  part_is = runtime->create_index_space(ctx, part_bounds);
  Realm::ZRect<3, coord_t> image_rect(Realm::ZPoint<3>(0, 0, 0),
                                      Realm::ZPoint<3>(width-1, height-1, num_images*3-1));
  IndexSpaceT<3> image_is = runtime->create_index_space(ctx, image_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  LogicalRegion image_lr = runtime->create_logical_region(ctx, image_is, fs);
  Realm::ZMatrix<3, 3, coord_t> transform;
  int extent_w = width / width_par;
  int extent_h = height / height_par;
  int extent_nc = 3 * num_images / image_par;
  Realm::ZRect<3, coord_t> extent(Realm::ZPoint<3>(0, 0, 0),
                                  Realm::ZPoint<3>(extent_w - 1, extent_h - 1, extent_nc - 1));
  transform[0][0] = extent_w; transform[0][1] = 0; transform[0][2] = 0;
  transform[1][0] = 0; transform[1][1] = extent_h; transform[1][2] = 0;
  transform[2][0] = 0; transform[2][1] = 0; transform[2][2] = extent_nc;
  IndexPartition image_ip = 
    runtime->create_partition_by_restriction(ctx, image_is, part_is, transform, extent);
  LogicalPartition image_lp = runtime->get_logical_partition(ctx, image_lr, image_ip);
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
  input_image.partition = image_lp;
};

