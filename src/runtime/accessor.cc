#include "flexflow/accessor.h"
#include "flexflow/model.h"

namespace FlexFlow {

using namespace Legion;

template <typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime) {
  const AccessorRO<DT, dim> acc(region, fid);
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
}

template <typename DT, int dim> TensorAccessorR<DT, dim>::TensorAccessorR() {}

GenericTensorAccessorR::GenericTensorAccessorR(int _num_dim,
                                               DataType _data_type,
                                               PhysicalRegion region,
                                               RegionRequirement req,
                                               FieldID fid,
                                               Context ctx,
                                               Runtime *runtime)
    : data_type(_data_type), ptr(NULL) {
  switch (_num_dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    if (data_type == DT_FLOAT) {                                               \
      TensorAccessorR<float, DIM> acc(region, req, fid, ctx, runtime);         \
      domain = acc.rect;                                                       \
      ptr = acc.ptr;                                                           \
    } else if (data_type == DT_DOUBLE) {                                       \
      TensorAccessorR<double, DIM> acc(region, req, fid, ctx, runtime);        \
      domain = acc.rect;                                                       \
      ptr = acc.ptr;                                                           \
    } else if (data_type == DT_INT64) {                                        \
      TensorAccessorR<int64_t, DIM> acc(region, req, fid, ctx, runtime);       \
      domain = acc.rect;                                                       \
      ptr = acc.ptr;                                                           \
    } else {                                                                   \
      assert(false && "Unsupported data_type");                                \
    }                                                                          \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  default: {
    fprintf(stderr, "Unsupported accessor dimension");
    assert(false);
  }
  }
}

template <typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime,
                                          bool readOutput) {
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  if (readOutput) {
    const AccessorRW<DT, dim> acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
  } else {
    const AccessorWO<DT, dim> acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
    // FIXME: currently we zero init the region if not read output
    // assign_kernel<DT><<<GET_BLOCKS(rect.volume()), CUDA_NUM_THREADS>>>(
    //    ptr, rect.volume(), 0.0f);
    // checkCUDA(cudaDeviceSynchronize());
  }
}

template <typename DT, int dim> TensorAccessorW<DT, dim>::TensorAccessorW() {}

GenericTensorAccessorW::GenericTensorAccessorW(int _num_dim,
                                               DataType _data_type,
                                               PhysicalRegion region,
                                               RegionRequirement req,
                                               FieldID fid,
                                               Context ctx,
                                               Runtime *runtime,
                                               bool readOutput)
    : data_type(_data_type), ptr(NULL) {
  switch (_num_dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    if (data_type == DT_FLOAT) {                                               \
      TensorAccessorW<float, DIM> acc(                                         \
          region, req, fid, ctx, runtime, readOutput);                         \
      domain = acc.rect;                                                       \
      ptr = acc.ptr;                                                           \
    } else if (data_type == DT_DOUBLE) {                                       \
      TensorAccessorW<double, DIM> acc(                                        \
          region, req, fid, ctx, runtime, readOutput);                         \
      domain = acc.rect;                                                       \
      ptr = acc.ptr;                                                           \
    } else if (data_type == DT_INT64) {                                        \
      TensorAccessorW<int64_t, DIM> acc(                                       \
          region, req, fid, ctx, runtime, readOutput);                         \
      domain = acc.rect;                                                       \
      ptr = acc.ptr;                                                           \
    } else {                                                                   \
      assert(false && "Unsupported data_type");                                \
    }                                                                          \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  default: {
    fprintf(stderr, "Unsupported accessor dimension");
    assert(false);
  }
  }
}

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

#define DIMFUNC(DIM)                                                           \
  template class TensorAccessorR<float, DIM>;                                  \
  template class TensorAccessorR<int32_t, DIM>;                                \
  template class TensorAccessorR<int64_t, DIM>;                                \
  template class TensorAccessorW<float, DIM>;                                  \
  template class TensorAccessorW<int32_t, DIM>;                                \
  template class TensorAccessorW<int64_t, DIM>;
LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC

template const float *helperGetTensorPointerRO(PhysicalRegion region,
                                               RegionRequirement req,
                                               FieldID fid,
                                               Context ctx,
                                               Runtime *runtime);
template float *helperGetTensorPointerRW(PhysicalRegion region,
                                         RegionRequirement req,
                                         FieldID fid,
                                         Context ctx,
                                         Runtime *runtime);
template float *helperGetTensorPointerWO(PhysicalRegion region,
                                         RegionRequirement req,
                                         FieldID fid,
                                         Context ctx,
                                         Runtime *runtime);

template const double *helperGetTensorPointerRO(PhysicalRegion region,
                                                RegionRequirement req,
                                                FieldID fid,
                                                Context ctx,
                                                Runtime *runtime);
template double *helperGetTensorPointerRW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime);
template double *helperGetTensorPointerWO(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime);

template const int32_t *helperGetTensorPointerRO(PhysicalRegion region,
                                                 RegionRequirement req,
                                                 FieldID fid,
                                                 Context ctx,
                                                 Runtime *runtime);
template int32_t *helperGetTensorPointerRW(PhysicalRegion region,
                                           RegionRequirement req,
                                           FieldID fid,
                                           Context ctx,
                                           Runtime *runtime);
template int32_t *helperGetTensorPointerWO(PhysicalRegion region,
                                           RegionRequirement req,
                                           FieldID fid,
                                           Context ctx,
                                           Runtime *runtime);

template const int64_t *helperGetTensorPointerRO(PhysicalRegion region,
                                                 RegionRequirement req,
                                                 FieldID fid,
                                                 Context ctx,
                                                 Runtime *runtime);
template int64_t *helperGetTensorPointerRW(PhysicalRegion region,
                                           RegionRequirement req,
                                           FieldID fid,
                                           Context ctx,
                                           Runtime *runtime);
template int64_t *helperGetTensorPointerWO(PhysicalRegion region,
                                           RegionRequirement req,
                                           FieldID fid,
                                           Context ctx,
                                           Runtime *runtime);

}; // namespace FlexFlow
