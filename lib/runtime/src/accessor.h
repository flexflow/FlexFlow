#ifndef _FF_ACCESSOR_H_
#define _FF_ACCESSOR_H_

#include "legion.h"
#include "kernels/ff_handler.h"
#include "kernels/accessor.h"
#include "op-attrs/ffconst.h"
#include "mappers/mapping_utilities.h"

using Legion::Mapping::Utilities::to_string;

namespace FlexFlow {

template <Legion::PrivilegeMode> struct privilege_mode_to_accessor_t { };

template <> struct privilege_mode_to_accessor_t<READ_WRITE> {
  using type = GenericTensorAccessorW;
};

template <> struct privilege_mode_to_accessor_t<READ_ONLY> {
  using type = GenericTensorAccessorR;
};

template <> struct privilege_mode_to_accessor_t<WRITE_ONLY> {
  using type = GenericTensorAccessorW;
};

template <Legion::PrivilegeMode PRIV>
using privilege_mode_to_accessor = typename privilege_mode_to_accessor_t<PRIV>::type;


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

template <typename Legion::PrivilegeMode PRIV>
privilege_mode_to_accessor<PRIV> 
helperGetGenericTensorAccessor(DataType datatype,
                               Legion::PhysicalRegion const &region,
                               Legion::RegionRequirement const &req, 
                               Legion::FieldID const &fid,
                               Legion::Context const &ctx, 
                               Legion::Runtime *runtime) {
  optional<variant<GenericTensorAccessorR, GenericTensorAccessorW>> result = nullopt;
  switch (PRIV) {
    case READ_ONLY:
      result = helperGetGenericTensorAccessorRO(datatype, region, req, fid, ctx, runtime);
    case WRITE_ONLY:
      result = helperGetGenericTensorAccessorWO(datatype, region, req, fid, ctx, runtime);
    case READ_WRITE:
      result = helperGetGenericTensorAccessorRW(datatype, region, req, fid, ctx, runtime);
    default:
      std::ostringstream oss;
      oss << "Unknown privilege mode " << to_string(PRIV);
      throw std::runtime_error(oss.str());
  }
  return get<privilege_mode_to_accessor<PRIV>>(result.value());
}

}

#endif
