#ifndef _FF_ACCESSOR_H_
#define _FF_ACCESSOR_H_
#include "ffconst.h"
#include "legion.h"
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

template <typename DT, int dim> struct TensorAccessorR {
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

template <typename DT, int dim> struct TensorAccessorW {
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

struct GenericTensorAccessorR {
  GenericTensorAccessorR(int num_dim,
                         DataType data_type,
                         Legion::PhysicalRegion region,
                         Legion::RegionRequirement req,
                         Legion::FieldID fid,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  GenericTensorAccessorR();
  DataType data_type;
  Legion::Domain domain;
  Legion::Memory memory;
  const void *ptr;
};

struct GenericTensorAccessorW {
  GenericTensorAccessorW(int num_dim,
                         DataType data_type,
                         Legion::PhysicalRegion region,
                         Legion::RegionRequirement req,
                         Legion::FieldID fid,
                         Legion::Context ctx,
                         Legion::Runtime *runtime,
                         bool readOutput = false);
  GenericTensorAccessorW();
  DataType data_type;
  Legion::Domain domain;
  Legion::Memory memory;
  void *ptr;
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
}; // namespace FlexFlow

#endif
