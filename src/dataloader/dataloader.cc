/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/dataloader.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace Legion;
using namespace FlexFlow;

SingleDataLoader::SingleDataLoader(FFModel &ff,
                                   ParallelTensor input,
                                   ParallelTensor full_input_,
                                   int num_samples_,
                                   DataType datatype_) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = num_samples_;
  datatype = datatype_;
  // Create full input
  assert(input->num_dims == full_input_->num_dims);
  for (int i = 0; i < input->num_dims - 1; i++) {
    assert(full_input_->dims[i].size == input->dims[i].size);
  }
  batch_input = input;
  // Currently assume that the leading dim of input is a replica dim of degree 1
  assert(input->dims[input->num_dims - 1].is_replica_dim);
  assert(input->dims[input->num_dims - 1].size == 1);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 1; i < input->num_dims - 1; i++) {
    dims[i - 1].size = input->dims[input->num_dims - 2 - i].size;
    dims[i - 1].degree = 1;
    dims[i - 1].parallel_idx = -1;
  }
  dims[0].size = num_samples;
  switch (input->num_dims - 1) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    full_input = ff.create_parallel_tensor<DIM>(dims, datatype);               \
    ff.map_tensor(full_input, NULL);                                           \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  int task_id = -1;
  if (datatype == DT_FLOAT) {
    task_id = PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID;
  } else if (datatype == DT_INT32) {
    task_id = PY_DL_INT32_LOAD_ENTIRE_CPU_TASK_ID;
  } else if (datatype == DT_INT64) {
    task_id = PY_DL_INT64_LOAD_ENTIRE_CPU_TASK_ID;
  } else {
    assert(0);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(task_id, TaskArgument(NULL, 0));
  // regions[0]: full_input
  launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_input->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[2]: full_input_
  launcher.add_region_requirement(RegionRequirement(
      full_input_->region, READ_ONLY, EXCLUSIVE, full_input_->region));
  launcher.add_field(1, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
  reset();
  next_batch(ff);
}

SingleDataLoader::SingleDataLoader(FFModel &ff,
                                   ParallelTensor input,
                                   void *full_input_ptr,
                                   int num_samples_,
                                   DataType datatype_) {
  num_samples = num_samples_;
  datatype = datatype_;
  // Currently assume that the leading dim of input is a replica dim of degree 1
  assert(input->dims[input->num_dims - 1].is_replica_dim);
  // assert(input->dims[input->num_dims - 1].size == 1);

  batch_input = input;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 1; i < input->num_dims; i++) {
    dims[i - 1].size = input->dims[input->num_dims - 1 - i].size;
    dims[i - 1].parallel_idx = -1;
    dims[i - 1].degree = 1;
  }
  dims[0].size = num_samples;

  int task_id = -1;
  if (datatype == DT_FLOAT) {
    task_id = PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID;
  } else if (datatype == DT_INT32) {
    task_id = PY_DL_INT32_INDEX_LOAD_ENTIRE_CPU_TASK_ID;
  } else if (datatype == DT_INT64) {
    task_id = PY_DL_INT64_INDEX_LOAD_ENTIRE_CPU_TASK_ID;
  } else {
    assert(0);
  }

  size_t size_per_sample = 1;
  for (int i = 1; i < input->num_dims - 1; i++) {
    assert(dims[i].size != 0);
    size_per_sample *= dims[i].size;
  }
  switch (input->num_dims - 1) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    full_input = ff.create_parallel_tensor<DIM>(dims, datatype);               \
    ff.map_tensor(full_input, NULL);                                           \
    index_loader_xd_launcher<DIM>(                                             \
        ff, task_id, full_input_ptr, size_per_sample);                         \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  reset();
  next_batch(ff);
}

template <int NDIM>
void SingleDataLoader::index_loader_xd_launcher(FFModel &ff,
                                                int task_id,
                                                void *full_input_ptr,
                                                size_t size_per_sample) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

#ifdef FF_PYTHON_USE_INDEX_LOADER
  IndexSpaceT<NDIM> task_is =
      IndexSpaceT<NDIM>(ff.get_or_create_task_is(NDIM, ""));
  Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
  ArgumentMap argmap;
  int total_shards = rect.volume();
  int idx = 0;
  for (PointInRectIterator<NDIM> it(rect); it(); it++) {
    IndexLoadArg meta;
    assert(num_samples % total_shards == 0);
    meta.num_samples = num_samples / total_shards;
    meta.size_per_sample = size_per_sample;
    meta.ptr = full_input_ptr;
    meta.idx = idx;
    argmap.set_point(*it, TaskArgument(&meta, sizeof(IndexLoadArg)));
    idx++;
  }
  // Load entire dataset
  IndexLauncher launcher(task_id, task_is, TaskArgument(NULL, 0), argmap);
  // regions[0]: full_input
  launcher.add_region_requirement(RegionRequirement(full_input->part,
                                                    0,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_input->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  FutureMap fu = runtime->execute_index_space(ctx, launcher);
  fu.wait_all_results();
#else
  IndexLoadArg meta;
  int total_shards = 1;
  assert(num_samples % total_shards == 0);
  meta.num_samples = num_samples / total_shards;
  meta.size_per_sample = size_per_sample;
  meta.ptr = full_input_ptr;
  meta.idx = 0;
  // Load entire dataset
  TaskLauncher launcher(task_id, TaskArgument(&meta, sizeof(IndexLoadArg)));
  // regions[0]: full_input
  launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_input->region,
                                                    MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  Future fu = runtime->execute_task(ctx, launcher);
  fu.wait();
#endif
}

void SingleDataLoader::reset() {
  next_index = 0;
}

void SingleDataLoader::next_batch(FFModel &ff) {
  int task_id = -1;
  if (datatype == DT_FLOAT) {
    task_id = PY_DL_FLOAT_LOAD_BATCH_GPU_TASK_ID;
  } else if (datatype == DT_INT32) {
    task_id = PY_DL_INT32_LOAD_BATCH_GPU_TASK_ID;
  } else if (datatype == DT_INT64) {
    task_id = PY_DL_INT64_LOAD_BATCH_GPU_TASK_ID;
  } else {
    assert(0);
  }
  switch (full_input->num_dims) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    next_batch_xd_launcher<DIM>(ff, task_id);                                  \
    break;
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template <int NDIM>
void SingleDataLoader::next_batch_xd_launcher(FFModel &ff, int task_id) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load input
#if 1
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize == batch_input->dims[NDIM - 1].size);
      meta.num_samples =
          batch_input->dims[NDIM - 1].size / batch_input->dims[NDIM - 1].degree;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(task_id,
                           batch_input->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_input->machine_view.hash());
    launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_input->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_input->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_input->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  next_index += ff.config.batchSize;
#else
  {
    IndexSpaceT<NDIM> task_is =
        IndexSpaceT<NDIM>(ff.get_or_create_task_is(NDIM, ""));
    Rect<NDIM> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    SampleIdxs meta;
    assert(ff.config.batchSize % (rect.hi[NDIM - 1] - rect.lo[NDIM - 1] + 1) ==
           0);
    meta.num_samples =
        ff.config.batchSize / (rect.hi[NDIM - 1] - rect.lo[NDIM - 1] + 1);
    for (int i = 0; i < meta.num_samples; i++) {
      meta.idxs[i] = idx++;
    }
    for (PointInRectIterator<NDIM> it(rect); it(); it++) {
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(task_id,
                           task_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           FFConfig::get_hash_id(""));
    launcher.add_region_requirement(RegionRequirement(full_input->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_input->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_input->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      batch_input->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
    next_index +=
        ff.config.batchSize / (rect.hi[NDIM - 1] - rect.lo[NDIM - 1] + 1);
  }
#endif
}

// Task body
template <typename DT>
void SingleDataLoader::load_entire_dataset_from_numpy(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return load_entire_dataset_from_numpy_with_dim<DT, DIM>(                   \
        task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template <typename DT, int NDIM>
void SingleDataLoader::load_entire_dataset_from_numpy_with_dim(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  AccessorWO<DT, NDIM> const acc_input(regions[0], FID_DATA);
  AccessorRO<DT, NDIM> const acc_input_(regions[1], FID_DATA);
  Rect<NDIM> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  Rect<NDIM> rect_input_ = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(acc_input_.accessor.is_dense_arbitrary(rect_input_));
  assert(rect_input_.volume() == rect_input.volume());

  DT *input_ptr = acc_input.ptr(rect_input.lo);
  const DT *input_ptr_ = acc_input_.ptr(rect_input_.lo);
  printf("Load entire dataset: ptr input_ %p %lu %lu, input %p %lu %lu\n",
         input_ptr_,
         (uintptr_t)input_ptr_,
         rect_input_.volume(),
         input_ptr,
         (uintptr_t)input_ptr,
         rect_input.volume());
  assert(rect_input.volume() == rect_input_.volume());
  memcpy(input_ptr, input_ptr_, sizeof(DT) * rect_input.volume());
  for (int i = 0; i < 32; i++) {
    std::cout << input_ptr[i] << " ";
  }
  std::cout << std::endl;
}

// Task body
template <typename DT>
void SingleDataLoader::index_load_entire_dataset_from_numpy(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == regions.size());
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return index_load_entire_dataset_from_numpy_with_dim<DT, DIM>(             \
        task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template <typename DT, int NDIM>
void SingleDataLoader::index_load_entire_dataset_from_numpy_with_dim(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == regions.size());
#ifdef FF_PYTHON_USE_INDEX_LOADER
  IndexLoadArg *meta = (IndexLoadArg *)task->local_args;
#else
  IndexLoadArg *meta = (IndexLoadArg *)task->args;
#endif
  AccessorWO<DT, NDIM> const acc_input(regions[0], FID_DATA);
  Rect<NDIM> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));

  DT *input_ptr = acc_input.ptr(rect_input.lo);
  size_t volume = meta->size_per_sample * meta->num_samples;
  DT *input_ptr_head_ = static_cast<DT *>(meta->ptr);
  DT *input_ptr_ = input_ptr_head_ + volume * meta->idx;

  printf("Index load entire dataset: ptr input_head_ %p, idx %d, input_ %p %lu "
         "%lu, input %p %lu %lu\n",
         input_ptr_head_,
         meta->idx,
         input_ptr_,
         (uintptr_t)input_ptr_,
         volume,
         input_ptr,
         (uintptr_t)input_ptr,
         rect_input.volume());
  assert(rect_input.volume() == volume);
  memcpy(input_ptr, input_ptr_, sizeof(DT) * volume);
  for (int i = 0; i < 32; i++) {
    std::cout << input_ptr[i] << " ";
  }
  std::cout << std::endl;
}

void SingleDataLoader::register_cpu_tasks(Runtime *runtime,
                                          bool pre_register,
                                          bool enable_control_replication) {
  if (!pre_register) {
    assert(runtime != NULL);
  }
  // float Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID,
                                   "Float Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<
          SingleDataLoader::load_entire_dataset_from_numpy<float>>(
          registrar, "Float Load Entire Dataset Task Numpy");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<
          SingleDataLoader::load_entire_dataset_from_numpy<float>>(registrar);
    }
  }
  // int32 Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(PY_DL_INT32_LOAD_ENTIRE_CPU_TASK_ID,
                                   "Int32 Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<
          SingleDataLoader::load_entire_dataset_from_numpy<int32_t>>(
          registrar, "Int32 Load Entire Dataset Task Numpy");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<
          SingleDataLoader::load_entire_dataset_from_numpy<int32_t>>(registrar);
    }
  }
  // int64 Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(PY_DL_INT64_LOAD_ENTIRE_CPU_TASK_ID,
                                   "Int64 Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<
          SingleDataLoader::load_entire_dataset_from_numpy<int64_t>>(
          registrar, "Int64 Load Entire Dataset Task Numpy");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<
          SingleDataLoader::load_entire_dataset_from_numpy<int64_t>>(registrar);
    }
  }
  // float Index load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
                                   "Float Index Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<
          SingleDataLoader::index_load_entire_dataset_from_numpy<float>>(
          registrar, "Float Index Load Entire Dataset Task Numpy");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<
          SingleDataLoader::index_load_entire_dataset_from_numpy<float>>(
          registrar);
    }
  }
  // int32 Index load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(PY_DL_INT32_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
                                   "Int32 Index Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<
          SingleDataLoader::index_load_entire_dataset_from_numpy<int32_t>>(
          registrar, "Int32 Index Load Entire Dataset Task Numpy");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<
          SingleDataLoader::index_load_entire_dataset_from_numpy<int32_t>>(
          registrar);
    }
  }
  // int64 Index load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(PY_DL_INT64_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
                                   "Int64 Index Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<
          SingleDataLoader::index_load_entire_dataset_from_numpy<int64_t>>(
          registrar, "Int64 Index Load Entire Dataset Task Numpy");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<
          SingleDataLoader::index_load_entire_dataset_from_numpy<int64_t>>(
          registrar);
    }
  }
}

void SingleDataLoader::register_gpu_tasks(Runtime *runtime,
                                          bool pre_register,
                                          bool enable_control_replication) {
  if (!pre_register) {
    assert(runtime != NULL);
  }
  // float load input
  {
    TaskVariantRegistrar registrar(PY_DL_FLOAT_LOAD_BATCH_GPU_TASK_ID,
                                   "Float Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<SingleDataLoader::load_input<float>>(
          registrar, "Float Load Input Task");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<SingleDataLoader::load_input<float>>(
          registrar);
    }
  }
  // int32 load input
  {
    TaskVariantRegistrar registrar(PY_DL_INT32_LOAD_BATCH_GPU_TASK_ID,
                                   "Int32 Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<SingleDataLoader::load_input<int32_t>>(
          registrar, "Int32 Load Input Task");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<SingleDataLoader::load_input<int32_t>>(
          registrar);
    }
  }
  // int64 load input
  {
    TaskVariantRegistrar registrar(PY_DL_INT64_LOAD_BATCH_GPU_TASK_ID,
                                   "Int64 Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<SingleDataLoader::load_input<int64_t>>(
          registrar, "Int64 Load Input Task");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<SingleDataLoader::load_input<int64_t>>(
          registrar);
    }
  }
}

template void SingleDataLoader::next_batch_xd_launcher<2>(FFModel &ff,
                                                          int task_id);
template void SingleDataLoader::next_batch_xd_launcher<4>(FFModel &ff,
                                                          int task_id);
template void SingleDataLoader::index_loader_xd_launcher<2>(
    FFModel &ff, int task_id, void *full_input_ptr, size_t size_per_sample);
template void SingleDataLoader::index_loader_xd_launcher<4>(
    FFModel &ff, int task_id, void *full_input_ptr, size_t size_per_sample);
template void SingleDataLoader::load_entire_dataset_from_numpy<float>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::load_entire_dataset_from_numpy<int32_t>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::load_entire_dataset_from_numpy<int64_t>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::index_load_entire_dataset_from_numpy<float>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::index_load_entire_dataset_from_numpy<int32_t>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
template void SingleDataLoader::index_load_entire_dataset_from_numpy<int64_t>(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime);
