#ifndef _FF_ACCESSOR_H_
#define _FF_ACCESSOR_H_
#include "ffconst.h"
#include "legion.h"

#if defined(FF_USE_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hip/hip_fp16.h>
#endif

// using namespace Legion;

namespace FlexFlow {

template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRO =
    Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRW = Legion::
    FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorWO = Legion::
    FieldAccessor<WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

template <typename DT, int dim>
struct TensorAccessorR {
  TensorAccessorR(Legion::PhysicalRegion region,
                  Legion::RegionRequirement req,
                  Legion::FieldID fid,
                  Legion::Context ctx,
                  Legion::Runtime *runtime);
  TensorAccessorR();
  Legion::Rect<dim> rect;
  Legion::Memory memory;
  const DT *ptr;
};

template <typename DT, int dim>
struct TensorAccessorW {
  TensorAccessorW(Legion::PhysicalRegion region,
                  Legion::RegionRequirement req,
                  Legion::FieldID fid,
                  Legion::Context ctx,
                  Legion::Runtime *runtime,
                  bool readOutput = false);
  TensorAccessorW();
  Legion::Rect<dim> rect;
  Legion::Memory memory;
  DT *ptr;
};

class GenericTensorAccessorW {
public:
  GenericTensorAccessorW();
  GenericTensorAccessorW(DataType data_type, Legion::Domain domain, void *ptr);
  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;
  char *get_byte_ptr() const;
  DataType data_type;
  Legion::Domain domain;
  void *ptr;
};

class GenericTensorAccessorR {
public:
  GenericTensorAccessorR();
  GenericTensorAccessorR(DataType data_type,
                         Legion::Domain domain,
                         void const *ptr);
  GenericTensorAccessorR(GenericTensorAccessorW const &acc);
  // GenericTensorAccessorR &operator=(GenericTensorAccessorW const &acc);
  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  half const *get_half_ptr() const;
  char const *get_byte_ptr() const;
  DataType data_type;
  Legion::Domain domain;
  void const *ptr;
};

template <typename DT>
const DT *helperGetTensorPointerRO(Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);

template <typename DT>
DT *helperGetTensorPointerWO(Legion::PhysicalRegion region,
                             Legion::RegionRequirement req,
                             Legion::FieldID fid,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

template <typename DT>
DT *helperGetTensorPointerRW(Legion::PhysicalRegion region,
                             Legion::RegionRequirement req,
                             Legion::FieldID fid,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

GenericTensorAccessorR
    helperGetGenericTensorAccessorRO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime);

GenericTensorAccessorW
    helperGetGenericTensorAccessorWO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime);

GenericTensorAccessorW
    helperGetGenericTensorAccessorRW(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime);

}; // namespace FlexFlow

#endif
