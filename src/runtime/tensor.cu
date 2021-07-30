#include "flexflow/tensor.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/config.h"
#include "flexflow/accessor.h"
#include "flexflow/model.h"

namespace FlexFlow {

using namespace Legion;

template <typename T>
bool TensorBase::set_tensor(
    const FFModel* ff,
    const std::vector<int>& dim_sizes,
    const T* data)
{
  Context ctx = ff->config.lg_ctx;
  Runtime* runtime = ff->config.lg_hlr;
  //TODO: check data type matches
  //TODO: Currently we use a task launch, change to index launch for NCCL parameter
  size_t volume = 1, num_replicas = 0;
  if (sync_type == ParameterSyncType::NCCL) {
    Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
    num_replicas = domain.get_volume();
  } else if (sync_type == ParameterSyncType::PS) {
    num_replicas = 1;
  } else {
    assert(false);
  }
  // Check dimensions
  if (num_dims != (int)dim_sizes.size())
    return false;
  for (int i = 0; i < num_dims; i++) {
    if (dims[num_dims-1-i].size != dim_sizes[i])
      return false;
    volume = volume * dim_sizes[i];
  }
  RegionRequirement req(region, READ_WRITE, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();
  switch (num_dims) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      TensorAccessorW<T, DIM> acc(pr, req, FID_DATA, ctx, runtime, true); \
      assert(acc.rect.volume() == volume * num_replicas); \
      T* ptr = acc.ptr; \
      for (size_t i = 0; i < num_replicas; i++) { \
        memcpy(ptr, data, volume * sizeof(T)); \
        ptr += volume; \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, pr);
  return true;
}

template <typename T>
bool TensorBase::get_tensor(
    const FFModel* ff,
    T* data)
{
  Context ctx = ff->config.lg_ctx;
  Runtime* runtime = ff->config.lg_hlr;
  LogicalRegion weight_lr = LogicalRegion::NO_REGION;
  if (sync_type == ParameterSyncType::PS) {
    weight_lr = region;
  } else {
    assert(owner_op != NULL);
    Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
      case DIM: \
      { \
        DomainPoint point = Point<DIM>::ZEROES(); \
        weight_lr = runtime->get_logical_subregion_by_color( \
            ctx, part, point); \
        break; \
      }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }
  }
  //TODO: check data type matches
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++) {
    volume = volume * dims[i].size;
  }
  RegionRequirement req(weight_lr, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();
  switch (num_dims) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      TensorAccessorR<T, DIM> acc(pr, req, FID_DATA, ctx, runtime); \
      assert(acc.rect.volume() == volume); \
      memcpy(data, acc.ptr, volume * sizeof(T)); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, pr);
  return true;
}

template bool TensorBase::set_tensor<float>(const FFModel* ff, const std::vector<int>& dims, const float* data);
template bool TensorBase::get_tensor<float>(const FFModel* ff, float* data);
template bool TensorBase::set_tensor<int>(const FFModel* ff, const std::vector<int>& dims, const int* data);
template bool TensorBase::get_tensor<int>(const FFModel* ff, int* data);
} // namespace FlexFlow
