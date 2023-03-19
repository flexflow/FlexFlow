/* Copyright 2022 NVIDIA CORPORATION
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

#ifndef __LEGION_TRITON_ACCESSOR_H__
#define __LEGION_TRITON_ACCESSOR_H__

#include "legion.h"
#include "types.h"

namespace triton { namespace backend { namespace legion {

template <Legion::PrivilegeMode MODE, int DIM>
class TensorAccessor {
 public:
  static inline void* access(
      DataType type, const Legion::Rect<DIM>& bounds,
      const Legion::PhysicalRegion& region)
  {
    // Legion doesn't understand types, it just knows about field
    // sizes so we just need to use types of the right size
    switch (sizeof_datatype(type)) {
      case 1: {
        Legion::FieldAccessor<
            MODE, int8_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int8_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      case 2: {
        Legion::FieldAccessor<
            MODE, int16_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int16_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      case 4: {
        Legion::FieldAccessor<
            MODE, int32_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int32_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      case 8: {
        Legion::FieldAccessor<
            MODE, int64_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int64_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      default:
        assert(false);
    }
    return nullptr;
  };
};

// specialization for read-only privileges to return a const void*
template <int DIM>
class TensorAccessor<LEGION_READ_ONLY, DIM> {
 public:
  static inline const void* access(
      DataType type, const Legion::Rect<DIM>& bounds,
      const Legion::PhysicalRegion& region)
  {
    // Legion doesn't understand types, it just knows about field
    // sizes so we just need to use types of the right size
    switch (sizeof_datatype(type)) {
      case 1: {
        Legion::FieldAccessor<
            LEGION_READ_ONLY, int8_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int8_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      case 2: {
        Legion::FieldAccessor<
            LEGION_READ_ONLY, int16_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int16_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      case 4: {
        Legion::FieldAccessor<
            LEGION_READ_ONLY, int32_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int32_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      case 8: {
        Legion::FieldAccessor<
            LEGION_READ_ONLY, int64_t, DIM, Legion::coord_t,
            Realm::AffineAccessor<int64_t, DIM, Legion::coord_t> >
            accessor(region, FID_DATA);
        return accessor.ptr(bounds);
      }
      default:
        assert(false);
    }
    return nullptr;
  };
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_ACCESSOR_H__
