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

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::concat(std::string name, int n, Tensor* tensors)
{
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  IndexSpaceT<3> task_is;
  Concat *cat = new Concat(name, config, n, tensors, task_is);
  layers.push_back(cat);
  return cat->output;
}

Concat::Concat(std::string _name, FFConfig _config,
               int _n, Tensor* _tensors, IndexSpaceT<3> _task_is)
 : Op(_name, _n, _tensors), task_is(_task_is), num_inputs(_n)
{
  Context ctx = _config.lg_ctx;
  HighLevelRuntime* runtime = _config.lg_hlr;
  FieldSpace fs = _config.field_space;

  Rect<3> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_n = part_rect.hi[2] - part_rect.lo[2] + 1;
  int input_w = inputs[0].adim[0];
  int input_h = inputs[0].adim[1];
  int input_c = 0;
  int input_n = inputs[0].adim[3];
  for (int i = 0; i < num_inputs; i++) {
    assert(input_w == inputs[i].adim[0]);
    assert(input_h == inputs[i].adim[1]);
    assert(input_n == inputs[i].adim[3]);
    input_c += inputs[i].adim[2];
  }
  int input_nc = input_n * input_c;
  Rect<3, coord_t> output_rect(Point<3>(0, 0, 0),
                      Point<3>(input_w-1, input_h-1, input_nc-1));
  IndexSpaceT<3> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr = runtime->create_logical_region(ctx, output_is, fs);
  Transform<3, 3, coord_t> transform;
  int extent_w = (input_w + num_par_w - 1) / num_par_w;
  int extent_h = (input_h + num_par_h - 1) / num_par_h;
  int extent_nc = input_nc / num_par_n;
  assert(input_nc % num_par_n == 0);
  Rect<3, coord_t> extent(Point<3>(0, 0, 0), Point<3>(extent_w-1, extent_h-1, extent_nc-1));
  transform[0][0] = extent_w; transform[0][1] = 0; transform[0][2] = 0;
  transform[1][0] = 0; transform[1][1] = extent_h; transform[1][2] = 0;
  transform[2][0] = 0; transform[2][1] = 0; transform[2][2] = extent_nc;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, task_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, output_ip));
  assert(runtime->is_index_partition_complete(ctx, output_ip));
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp =
    runtime->get_logical_partition(ctx, output_grad_lr, output_ip);

  output.numDim = 4;
  output.adim[0] = input_w;
  output.adim[1] = input_h;
  output.adim[2] = input_c;
  output.adim[3] = inputs[0].adim[3];
  output.pdim[0] = extent_w;
  output.pdim[1] = extent_h;
  output.pdim[2] = input_c;
  output.pdim[3] = extent_nc / input_c;
  assert(extent_nc % input_c == 0);
  output.region = output_lr;
  output.part = output_lp;
  output.region_grad = output_grad_lr;
  output.part_grad = output_grad_lp;
  printf("Create concat layer: output(n=%d c=%d h=%d w=%d)\n",
         output.adim[3], output.adim[2], output.adim[1], output.adim[0]);
  for (int i = 0; i < num_inputs; i++) {
    // For now, we assume our output has the same partition as all inputs
    Rect<3> input_part_rect =
      runtime->get_index_partition_color_space(ctx, inputs[i].part.get_index_partition());
    assert(part_rect == input_part_rect);
    input_lps[i] = inputs[i].part;
  }
  return;
}

__host__
OpMeta* Concat::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  ConcatMeta* m = new ConcatMeta(handler);
  return m;
}

void Concat::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    FFHandler handler = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handler, sizeof(FFHandler)));
  }
  IndexLauncher init_launcher(CONCAT_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(Concat)), argmap);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](O): output
  regions[1..num_inputs](I): inputs
*/
void Concat::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  assert(regions.size() == cc->num_inputs + 1);
  assert(task->regions.size() == cc->num_inputs + 1);
  const AccessorWO<float, 3> acc_output(regions[0], FID_DATA);
  Rect<3> rect_output;
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  float *output_ptr = acc_output.ptr(rect_output.lo);
  float *output_bound = output_ptr + rect_output.volume();
  for (int i = 0; i < cc->num_inputs; i++) {
    const AccessorRO<float, 3> acc_input(regions[i+1], FID_DATA);
    Rect<3> rect_input =
      runtime->get_index_space_domain(ctx, task->regions[i+1].region.get_index_space());
    assert(acc_input.accessor.is_dense_arbitrary(rect_input));
    const float *input_ptr = acc_input.ptr(rect_input.lo);
    checkCUDA(cudaMemcpyAsync(output_ptr, input_ptr,
                              rect_input.volume() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    output_ptr += rect_input.volume();
  }
  assert(output_ptr == output_bound);
}

void Concat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(CONCAT_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Concat)), argmap);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < num_inputs; i++) {
    launcher.add_region_requirement(
        RegionRequirement(inputs[i].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): output_grad
  regions[1..num_inputs](O): input_grad
*/
void Concat::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  assert(regions.size() == cc->num_inputs + 1);
  assert(task->regions.size() == cc->num_inputs + 1);
  const AccessorRO<float, 3> acc_output(regions[0], FID_DATA);
  Rect<3> rect_output;
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  float *output_ptr = (float*) acc_output.ptr(rect_output.lo);
  float *output_bound = output_ptr + rect_output.volume();
  for (int i = 0; i < cc->num_inputs; i++) {
    const AccessorWO<float, 3> acc_input(regions[i+1], FID_DATA);
    Rect<3> rect_input =
      runtime->get_index_space_domain(ctx, task->regions[i+1].region.get_index_space());
    assert(acc_input.accessor.is_dense_arbitrary(rect_input));
    float *input_ptr = acc_input.ptr(rect_input.lo);
    checkCUDA(cudaMemcpyAsync(input_ptr, output_ptr,
                              rect_input.volume() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    output_ptr += rect_input.volume();
  }
  assert(output_ptr == output_bound);
}

void Concat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(CONCAT_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Concat)), argmap);
  launcher.add_region_requirement(
      RegionRequirement(output.part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < num_inputs; i++) {
    launcher.add_region_requirement(
        RegionRequirement(inputs[i].part_grad, 0/*projection id*/,
                          WRITE_DISCARD, EXCLUSIVE, inputs[i].region_grad));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Concat::update(const FFModel& ff)
{
}
