#include "accessor.h"
#include "kernels/accessor.h"
#include "kernels/datatype_dispatch.h"
#include "legion.h"

namespace FlexFlow {

using namespace Legion;

template <typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime) {
  AccessorRO<DT, dim> const acc(region, fid);
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
}

template <typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR() {}

template <typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime,
                                          bool readOutput) {
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  if (readOutput) {
    AccessorRW<DT, dim> const acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
  } else {
    AccessorWO<DT, dim> const acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
    // FIXME: currently we zero init the region if not read output
    // assign_kernel<DT><<<GET_BLOCKS(rect.volume()), CUDA_NUM_THREADS>>>(
    //    ptr, rect.volume(), 0.0f);
    // checkCUDA(cudaDeviceSynchronize());
  }
}

template <typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW() {}

template <typename DT>
const DT *helperGetTensorPointerRO(PhysicalRegion region,
                                   RegionRequirement req,
                                   FieldID fid,
                                   Context ctx,
                                   Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<DT, DIM> acc(region, req, fid, ctx, runtime);              \
    return acc.ptr;                                                            \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template <typename DT>
DT *helperGetTensorPointerRW(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<DT, DIM> acc(                                              \
        region, req, fid, ctx, runtime, true /*readOutput*/);                  \
    return acc.ptr;                                                            \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template <typename DT>
DT *helperGetTensorPointerWO(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<DT, DIM> acc(                                              \
        region, req, fid, ctx, runtime, false /*readOutput*/);                 \
    return acc.ptr;                                                            \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template <DataType DT>
struct GetTensorPointerWOFunctor {
  void *operator()(PhysicalRegion region,
                   RegionRequirement req,
                   FieldID fid,
                   Context ctx,
                   Runtime *runtime) const {
    return (void *)helperGetTensorPointerWO<real_type_t<DT>>(
        region, req, fid, ctx, runtime);
  }
};

template <DataType DT>
struct GetTensorPointerROFunctor {
  void const *operator()(PhysicalRegion region,
                         RegionRequirement req,
                         FieldID fid,
                         Context ctx,
                         Runtime *runtime) const {
    return (void const *)helperGetTensorPointerRO<real_type_t<DT>>(
        region, req, fid, ctx, runtime);
  }
};

template <DataType DT>
struct GetTensorPointerRWFUnctor {
  void *operator()(PhysicalRegion region,
                   RegionRequirement req,
                   FieldID fid,
                   Context ctx,
                   Runtime *runtime) const {
    return (void *)helperGetTensorPointerRW<real_type_t<DT>>(
        region, req, fid, ctx, runtime);
  }
};

static ArrayShape to_array_shape(Legion::Domain const &domain) {
  if (domain == Domain::NO_DOMAIN) {
    throw std::runtime_error("Cannot convert domain NO_DOMAIN");
  }

  std::vector<std::size_t> dimension_sizes;
  int num_dims = domain.get_dim();
  for (int i = 0; i < num_dims; i++) {
    dimension_sizes.push_back(domain.lo()[i] - domain.hi()[i]);
  }

  return {dimension_sizes};
}

GenericTensorAccessorR getGenericTensorAccessorRO(DataType datatype,
                                                  Legion::PhysicalRegion region,
                                                  Legion::RegionRequirement req,
                                                  Legion::FieldID fid,
                                                  Legion::Context ctx,
                                                  Legion::Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void const *ptr = DataTypeDispatch1<GetTensorPointerROFunctor>{}(
      datatype, region, req, fid, ctx, runtime);
  return {datatype, to_array_shape(domain), ptr};
}

GenericTensorAccessorW
    helperGetGenericTensorAccessorWO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime) {

  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void *ptr = DataTypeDispatch1<GetTensorPointerWOFunctor>{}(
      datatype, region, req, fid, ctx, runtime);
  return {datatype, to_array_shape(domain), ptr};
}

GenericTensorAccessorW
    helperGetGenericTensorAccessorRW(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void *ptr = DataTypeDispatch1<GetTensorPointerRWFUnctor>{}(
      datatype, region, req, fid, ctx, runtime);
  return {datatype, to_array_shape(domain), ptr};
}

} // namespace FlexFlow
