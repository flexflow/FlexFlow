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
                   int fc_par_n, int fc_par_c,
                   Context ctx, Runtime* runtime)
{
  config.lg_ctx = ctx;
  config.lg_hlr = runtime;
  config.num_par_w = width_par;
  config.num_par_h = height_par;
  config.num_par_n = image_par;
  config.num_workers = width_par * height_par * image_par;
  config.fc_num_par_c = fc_par_c;
  config.fc_num_par_n = fc_par_n;
  Realm::ZRect<3, coord_t> part_bounds(Realm::ZPoint<3>(0, 0, 0),
                             Realm::ZPoint<3>(width_par-1, height_par-1, image_par-1));
  part_is = runtime->create_index_space(ctx, part_bounds);
  Realm::ZRect<2, coord_t> fc_part_bounds(Realm::ZPoint<2>(0, 0),
                             Realm::ZPoint<2>(fc_par_c-1, fc_par_n-1));
  fc_part_is = runtime->create_index_space(ctx, fc_part_bounds);
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
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }
  
  int output_c = input.adim[0] * input.adim[1] * input.adim[2];
  int output_n = input.adim[3];
  Realm::ZRect<2, coord_t> output_rect(Realm::ZPoint<2>(0, 0),
                     Realm::ZPoint<2>(output_c-1, output_n-1));
  IndexSpaceT<2> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Realm::ZMatrix<2, 2, coord_t> transform;
  int extent_c = input.pdim[0] * input.pdim[1] * input.pdim[2];
  int extent_n = input.pdim[3];
  Realm::ZRect<2, coord_t> extent(Realm::ZPoint<2>(0, 0),
                     Realm::ZPoint<2>(extent_c-1,extent_n-1));
  transform[0][0] = extent_c; transform[0][1] = 0;
  transform[1][0] = 0; transform[1][1] = extent_n;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is_2d, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  output.numDim = 2;
  output.adim[0] = output_c;
  output.adim[1] = output_n;
  output.pdim[0] = extent_c;
  output.pdim[1] = extent_n;
  output.region = output_lr;
  output.partition = output_lp;
  printf("flat: input(N=%d C=%d H=%d W=%d) -> output(N=%d C=%d)\n",
         input.pdim[3], input.pdim[2], input.pdim[1], input.pdim[0], output.pdim[1], output.pdim[0]);
 
  Realm::ZMatrix<2, 3, coord_t> flat_trans;
  flat_trans[0][0] = input.pdim[0] * input.pdim[1] * input.adim[2];
  flat_trans[0][1] = input.adim[0] * input.pdim[1] * input.adim[2];
  flat_trans[0][2] = 0;
  flat_trans[1][0] = 0;
  flat_trans[1][1] = 0;
  flat_trans[1][2] = input.pdim[3];
  IndexPartition flat_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is_3d, flat_trans, extent);
  flat_lp = runtime->get_logical_partition(ctx, output_lr, flat_ip);
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
  Realm::ZRect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
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
  regions[1](I): output
*/  
void Flat::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 4);
  const FieldAccessor<READ_ONLY, float, 3> acc_input(regions[0], FID_DATA);
  const FieldAccessor<WRITE_DISCARD, float, 2> acc_output(regions[1], FID_DATA);
  Realm::ZRect<3> rect_input;
  Realm::ZRect<2> rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  assert(rect_input.volume() == rect_output.volume());

  checkCUDA(cudaMemcpy(output_ptr, input_ptr,
                       rect_input.volume() * sizeof(float),
                       cudaMemcpyDeviceToDevice));
}

void Flat::forward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Realm::ZRect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
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
  regions[0](O) : input
  regions[1](I) : output
*/
void Flat::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
}

void Flat::backward(const CnnModel& model)
{
}
