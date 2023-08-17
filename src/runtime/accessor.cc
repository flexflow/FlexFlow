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
  AccessorRO<DT, dim> const acc(region, fid);
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
}

template <typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR() {}

GenericTensorAccessorR::GenericTensorAccessorR(DataType _data_type,
                                               Legion::Domain _domain,
                                               void const *_ptr)
    : data_type(_data_type), domain(_domain), ptr(_ptr) {}

GenericTensorAccessorR::GenericTensorAccessorR(
    GenericTensorAccessorW const &acc)
    : data_type(acc.data_type), domain(acc.domain), ptr(acc.ptr) {}

GenericTensorAccessorR::GenericTensorAccessorR()
    : data_type(DT_NONE), domain(Domain::NO_DOMAIN), ptr(nullptr) {}

int32_t const *GenericTensorAccessorR::get_int32_ptr() const {
  if (data_type == DT_INT32) {
    return static_cast<int32_t const *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<int32_t const *>(nullptr);
  }
}

int64_t const *GenericTensorAccessorR::get_int64_ptr() const {
  if (data_type == DT_INT64) {
    return static_cast<int64_t const *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<int64_t const *>(nullptr);
  }
}

float const *GenericTensorAccessorR::get_float_ptr() const {
  if (data_type == DT_FLOAT) {
    return static_cast<float const *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<float const *>(nullptr);
  }
}

double const *GenericTensorAccessorR::get_double_ptr() const {
  if (data_type == DT_DOUBLE) {
    return static_cast<double const *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<double const *>(nullptr);
  }
}

half const *GenericTensorAccessorR::get_half_ptr() const {
  if (data_type == DT_HALF) {
    return static_cast<half const *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<half const *>(nullptr);
  }
}

char const *GenericTensorAccessorR::get_byte_ptr() const {
  if (data_type == DT_INT4 || data_type == DT_INT8) {
    return static_cast<char const *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<char const *>(nullptr);
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

GenericTensorAccessorW::GenericTensorAccessorW(DataType _data_type,
                                               Legion::Domain _domain,
                                               void *_ptr)
    : data_type(_data_type), domain(_domain), ptr(_ptr) {}

GenericTensorAccessorW::GenericTensorAccessorW()
    : data_type(DT_NONE), domain(Domain::NO_DOMAIN), ptr(nullptr) {}

int32_t *GenericTensorAccessorW::get_int32_ptr() const {
  if (data_type == DT_INT32) {
    return static_cast<int32_t *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<int32_t *>(nullptr);
  }
}

int64_t *GenericTensorAccessorW::get_int64_ptr() const {
  if (data_type == DT_INT64) {
    return static_cast<int64_t *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<int64_t *>(nullptr);
  }
}

float *GenericTensorAccessorW::get_float_ptr() const {
  if (data_type == DT_FLOAT) {
    return static_cast<float *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<float *>(nullptr);
  }
}

double *GenericTensorAccessorW::get_double_ptr() const {
  if (data_type == DT_DOUBLE) {
    return static_cast<double *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<double *>(nullptr);
  }
}

half *GenericTensorAccessorW::get_half_ptr() const {
  if (data_type == DT_HALF) {
    return static_cast<half *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<half *>(nullptr);
  }
}

char *GenericTensorAccessorW::get_byte_ptr() const {
  if (data_type == DT_INT4 || data_type == DT_INT8) {
    return static_cast<char *>(ptr);
  } else {
    assert(false && "Invalid Accessor Type");
    return static_cast<char *>(nullptr);
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

GenericTensorAccessorR
    helperGetGenericTensorAccessorRO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void const *ptr = nullptr;
  switch (datatype) {
    case DT_INT32: {
      ptr = helperGetTensorPointerRO<int32_t>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT64: {
      ptr = helperGetTensorPointerRO<int64_t>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_HALF: {
      ptr = helperGetTensorPointerRO<half>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_FLOAT: {
      ptr = helperGetTensorPointerRO<float>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_DOUBLE: {
      ptr = helperGetTensorPointerRO<double>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT4: {
      ptr = helperGetTensorPointerRO<char>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT8: {
      ptr = helperGetTensorPointerRO<char>(region, req, fid, ctx, runtime);
      break;
    }
    default: {
      assert(false);
    }
  }
  return GenericTensorAccessorR(datatype, domain, ptr);
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
  void *ptr = nullptr;
  switch (datatype) {
    case DT_INT32: {
      ptr = helperGetTensorPointerWO<int32_t>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT64: {
      ptr = helperGetTensorPointerWO<int64_t>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_HALF: {
      ptr = helperGetTensorPointerWO<half>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_FLOAT: {
      ptr = helperGetTensorPointerWO<float>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_DOUBLE: {
      ptr = helperGetTensorPointerWO<double>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT4: {
      ptr = helperGetTensorPointerWO<char>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT8: {
      ptr = helperGetTensorPointerWO<char>(region, req, fid, ctx, runtime);
      break;
    }
    default: {
      assert(false);
    }
  }
  return GenericTensorAccessorW(datatype, domain, ptr);
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
  void *ptr = nullptr;
  switch (datatype) {
    case DT_INT32: {
      ptr = helperGetTensorPointerRW<int32_t>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT64: {
      ptr = helperGetTensorPointerRW<int64_t>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_HALF: {
      ptr = helperGetTensorPointerRW<half>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_FLOAT: {
      ptr = helperGetTensorPointerRW<float>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_DOUBLE: {
      ptr = helperGetTensorPointerRW<double>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT4: {
      ptr = helperGetTensorPointerRW<char>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_INT8: {
      ptr = helperGetTensorPointerRW<char>(region, req, fid, ctx, runtime);
      break;
    }
    default: {
      assert(false);
    }
  }
  return GenericTensorAccessorW(datatype, domain, ptr);
}

#define DIMFUNC(DIM)                                                           \
  template class TensorAccessorR<char, DIM>;                                   \
  template class TensorAccessorR<half, DIM>;                                   \
  template class TensorAccessorR<float, DIM>;                                  \
  template class TensorAccessorR<double, DIM>;                                 \
  template class TensorAccessorR<int32_t, DIM>;                                \
  template class TensorAccessorR<int64_t, DIM>;                                \
  template class TensorAccessorW<char, DIM>;                                   \
  template class TensorAccessorW<half, DIM>;                                   \
  template class TensorAccessorW<float, DIM>;                                  \
  template class TensorAccessorW<double, DIM>;                                 \
  template class TensorAccessorW<int32_t, DIM>;                                \
  template class TensorAccessorW<int64_t, DIM>;
LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
template half const *helperGetTensorPointerRO(PhysicalRegion region,
                                              RegionRequirement req,
                                              FieldID fid,
                                              Context ctx,
                                              Runtime *runtime);
template half *helperGetTensorPointerRW(PhysicalRegion region,
                                        RegionRequirement req,
                                        FieldID fid,
                                        Context ctx,
                                        Runtime *runtime);
template half *helperGetTensorPointerWO(PhysicalRegion region,
                                        RegionRequirement req,
                                        FieldID fid,
                                        Context ctx,
                                        Runtime *runtime);

template char const *helperGetTensorPointerRO(PhysicalRegion region,
                                              RegionRequirement req,
                                              FieldID fid,
                                              Context ctx,
                                              Runtime *runtime);
template char *helperGetTensorPointerRW(PhysicalRegion region,
                                        RegionRequirement req,
                                        FieldID fid,
                                        Context ctx,
                                        Runtime *runtime);
template char *helperGetTensorPointerWO(PhysicalRegion region,
                                        RegionRequirement req,
                                        FieldID fid,
                                        Context ctx,
                                        Runtime *runtime);

template float const *helperGetTensorPointerRO(PhysicalRegion region,
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

template double const *helperGetTensorPointerRO(PhysicalRegion region,
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

template int32_t const *helperGetTensorPointerRO(PhysicalRegion region,
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

template int64_t const *helperGetTensorPointerRO(PhysicalRegion region,
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
