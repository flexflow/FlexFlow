#ifndef _FF_ACCESSOR_H_
#define _FF_ACCESSOR_H_
#include "legion.h"
using namespace Legion;

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

template<typename DT, int dim>
struct TensorAccessorR {
  TensorAccessorR(PhysicalRegion region,
                  RegionRequirement req,
                  FieldID fid,
                  Context ctx,
                  Runtime* runtime);
  TensorAccessorR();
  Rect<dim> rect;
  Memory memory;
  const DT *ptr;
};

template<typename DT, int dim>
struct TensorAccessorW {
  TensorAccessorW(PhysicalRegion region,
                  RegionRequirement req,
                  FieldID fid,
                  Context ctx,
                  Runtime* runtime,
                  bool readOutput = false);
  TensorAccessorW();
  Rect<dim> rect;
  Memory memory;
  DT *ptr;
};

template<typename DT>
const DT* helperGetTensorPointerRO(PhysicalRegion region,
                                  RegionRequirement req,
                                  FieldID fid,
                                  Context ctx,
                                  Runtime* runtime);

template<typename DT>
DT* helperGetTensorPointerWO(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime* runtime);

template<typename DT>
DT* helperGetTensorPointerRW(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime* runtime);
#endif
