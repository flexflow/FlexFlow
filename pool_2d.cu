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
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>

Tensor CnnModel::add_pooling_layer(Tensor input,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w)
{
  assert(input.numDim == 4); /*NCHW*/
  Pooling2D *pool = new Pooling2D(config, input, part_is, kernel_h, kernel_w,
                                  stride_h, stride_w, padding_h, padding_w);
  layers.push_back(pool);
  return pool->output;
}

Pooling2D::Pooling2D(CnnConfig config, Tensor input, IndexSpaceT<3> part_is,
                     int _kernel_h, int _kernel_w, int _stride_h, int _stride_w,
                     int _padding_h, int _padding_w)
: Op(input), kernel_h(_kernel_h), kernel_w(_kernel_w), stride_h(_stride_h),
  stride_w(_stride_w), padding_h(_padding_h), padding_w(_padding_w)
{
  Context ctx = config.lg_ctx;
  HighLevelRuntime* runtime = config.lg_hlr;

  int input_w = input.adim[0];
  int input_h = input.adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_nc = input.adim[3] * input.adim[2];
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(float), FID_DATA);
  }

  Realm::ZRect<3, coord_t> output_rect(Realm::ZPoint<3>(0, 0, 0),
                      Realm::ZPoint<3>(output_w-1, output_h-1, output_nc-1));
  IndexSpaceT<3> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  Realm::ZMatrix<3, 3, coord_t> transform;
  int extent_w = (output_w + config.num_par_w - 1) / config.num_par_w;
  int extent_h = (output_h + config.num_par_h - 1) / config.num_par_h;
  int extent_nc = output_nc / config.num_par_n;
  assert(output_nc % config.num_par_n == 0);
  Realm::ZRect<3, coord_t> extent(Realm::ZPoint<3>(0, 0, 0),
                      Realm::ZPoint<3>(extent_w-1, extent_h-1, extent_nc-1));
  transform[0][0] = extent_w; transform[0][1] = 0; transform[0][2] = 0;
  transform[1][0] = 0; transform[1][1] = extent_h; transform[1][2] = 0;
  transform[2][0] = 0; transform[2][1] = 0; transform[2][2] = extent_nc;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, part_is, transform, extent);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);

  output.numDim = 4;
  output.adim[0] = output_w;
  output.adim[1] = output_h;
  output.adim[2] = input.adim[2];
  output.adim[3] = input.adim[3];
  output.pdim[0] = extent_w;
  output.pdim[1] = extent_h;
  output.pdim[2] = output.adim[2];
  output.pdim[3] = output.adim[3];
  output.region = output_lr;
  output.partition = output_lp;
}

/*
*/
OpMeta* Pooling2D::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  const Pooling2D* pool = (Pooling2D*) task->args;
  assert(regions.size() == 0);
  assert(regions.size() == 0);
  CnnHandle handle = *((const CnnHandle*) task->local_args);
  Pooling2DMeta* m = new Pooling2DMeta(handle);
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&m->poolDesc));

  printf("pool(inputDim): n(%d) c(%d) h(%d) w(%d)\n", pool->inputs[0].pdim[3],
        pool->inputs[0].pdim[2], pool->inputs[0].pdim[1], pool->inputs[0].pdim[0]);
  printf("pool(outputDim): n(%d) c(%d) h(%d) w(%d)\n", pool->output.pdim[3],
        pool->output.pdim[2], pool->output.pdim[1], pool->output.pdim[0]);
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        pool->inputs[0].pdim[3],
                                        pool->inputs[0].pdim[2],
                                        pool->inputs[0].pdim[1],
                                        pool->inputs[0].pdim[0]));

  checkCUDNN(cudnnSetPooling2dDescriptor(m->poolDesc,
                                         CUDNN_POOLING_MAX,
                                         CUDNN_PROPAGATE_NAN,
                                         pool->kernel_h,
                                         pool->kernel_w,
                                         pool->padding_h,
                                         pool->padding_w,
                                         pool->stride_h,
                                         pool->stride_w));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(m->poolDesc,
                                               m->inputTensor,
                                               &n, &c, &h, &w));
  assert(n == pool->output.pdim[3]);
  assert(c == pool->output.pdim[2]);
  assert(h == pool->output.pdim[1]);
  assert(w == pool->output.pdim[0]);


  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  return m;
}

void Pooling2D::init(const CnnModel& model)
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
  IndexLauncher init_launcher(POOL2D_INIT_TASK_ID, model.part_is,
                              TaskArgument(this, sizeof(Pooling2D)), argmap);
  idx = 0;
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  for (PointInRectIterator<3> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/

void Pooling2D::forward_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  const Pooling2DMeta* m = *((Pooling2DMeta**) task->local_args);
  const FieldAccessor<READ_ONLY, float, 3> acc_input(regions[0], FID_DATA);
  const FieldAccessor<WRITE_DISCARD, float, 3> acc_output(regions[1], FID_DATA);
  Realm::ZRect<3> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);

  checkCUDNN(cudnnPoolingForward(m->handle.dnn, m->poolDesc,
                                 &alpha, m->inputTensor, input_ptr,
                                 &beta, m->outputTensor, output_ptr));
}

void Pooling2D::forward(const CnnModel& model)
{
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Realm::ZRect<3> rect = runtime->get_index_space_domain(ctx, model.part_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    printf("mp.pointer = %llx\n", mp);
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(POOL2D_FWD_TASK_ID, model.part_is,
                         TaskArgument(NULL, 0), argmap);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].partition, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.partition, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Pooling2D::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
}

void Pooling2D::backward(const CnnModel& model)
{
}
