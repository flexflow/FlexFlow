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
#include "cuda_helper.h"

Tensor FFModel::flat(std::string name, Tensor input)
{
  assert(input.numDim == 4);
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  Flat *flat = new Flat(*this, name, input);
  flat->add_to_model(*this);
  return flat->output;
}

Flat* FFModel::flat(std::string name)
{
  //assert(strategies.find(name) != strategies.end());
  //ParallelConfig pc = strategies[name];
  Flat *flat = new Flat(*this, name);
  return flat;
}

Flat::Flat(FFModel& model,
           const std::string& pcname,
           const Tensor& _input)
: Op(pcname, _input)
{
  create_output_and_partition(model);
#ifdef DEADCODE
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
    runtime->create_partition_by_restriction(ctx, output_is, task_is_2d, transform, extent);
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
  output.part = output_lp;
  output.part_grad = output_grad_lp;
  printf("Create flat layer: input(N=%d C=%d H=%d W=%d) -> output(N=%d C=%d)\n",
         _input.adim[3], _input.adim[2], _input.adim[1], _input.adim[0],
         output.adim[1], output.adim[0]);
 
  FieldSpace proj_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, proj_fs);
    allocator.allocate_field(sizeof(Rect<2>), FID_DATA);
  }
  LogicalRegion proj_lr = runtime->create_logical_region(ctx, task_is_3d, proj_fs);
  InlineLauncher launcher(
      RegionRequirement(proj_lr, WRITE_DISCARD, EXCLUSIVE, proj_lr)
                                           .add_field(FID_DATA));
  PhysicalRegion proj_pr = runtime->map_region(ctx, launcher);
  proj_pr.wait_until_valid();
  coord_t subtotal = 0;
  {
    const FieldAccessor<WRITE_DISCARD, Rect<2>, 3, coord_t,
              Realm::AffineAccessor<Rect<2>, 3, coord_t> > ra(proj_pr, FID_DATA);
    Rect<3> rect = runtime->get_index_space_domain(ctx, task_is_3d);
    for(PointInRectIterator<3> pir(rect); pir(); ++pir) {
      IndexSpace subspace = runtime->get_index_subspace(_input.part.get_index_partition(), *pir);
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
    runtime->create_partition_by_restriction(ctx, task_is_3d, task_is_3d,
                                             proj_trans, proj_extent);
  LogicalPartition proj_lp =
     runtime->get_logical_partition(ctx, proj_lr, proj_ip);
  IndexPartition flat_ip =
    runtime->create_partition_by_image_range(ctx, output_is,
                         proj_lp, proj_lr, FID_DATA, task_is_3d);
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
#endif
}

Flat::Flat(FFModel& model,
           const std::string& pcname)
: Op(pcname)
{
}

Tensor Flat::init_inout(FFModel& model, const Tensor& _input)
{
  add_to_model(model);
  inputs[0] = _input;
  create_output_and_partition(model);
  return output;
}

void Flat::add_to_model(FFModel& model)
{
  model.layers.push_back(this);
}

void Flat::create_output_and_partition(FFModel& model)
{
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
  // Assert data parallelism for operators with dim changes
  assert(num_par_c == 1);
 
  int out_dim = inputs[0].adim[0] * inputs[0].adim[1] * inputs[0].adim[2];
  int batch_size = inputs[0].adim[3];
  // Create output tensor
  {
    const int dims[2] = {batch_size, out_dim};
    output = model.create_tensor<2>(dims, task_is, DT_FLOAT);
  }
  model.create_data_parallel_partition_with_diff_dims<4, 2>(
      inputs[0], task_is, input_lps[0], input_grad_lps[0]);
}

OpMeta* Flat::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  FlatMeta* m = new FlatMeta(handler);
  return m;
}

void Flat::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }

  IndexLauncher launcher(FLAT_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Flat)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
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
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, 4> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);

  assert(acc_input.rect.volume() == acc_output.rect.volume());
  checkCUDA(cudaMemcpyAsync(acc_output.ptr, acc_input.ptr,
                            acc_input.rect.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
}

void Flat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(FLAT_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, output.region));
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
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, 4> acc_input_grad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  TensorAccessorR<float, 2> acc_output_grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());

  checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
                            acc_input_grad.rect.volume() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
}

void Flat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(FLAT_BWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

