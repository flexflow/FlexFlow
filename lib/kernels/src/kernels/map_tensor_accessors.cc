#include "kernels/map_tensor_accessors.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

struct F1 {
  template <typename T>
  float operator()(T const &t) const {
    PANIC();
  }
};

template GenericTensorAccessorW
    map_tensor_accessor(GenericTensorAccessorR const &, F1 &&, Allocator &);

struct F2 {
  template <typename T1, typename T2>
  float operator()(T1 const &lhs, T2 const &rhs) const {
    PANIC();
  }
};

template GenericTensorAccessorW
    map_tensor_accessors2(GenericTensorAccessorR const &,
                          GenericTensorAccessorR const &,
                          DataType,
                          F2 &&,
                          Allocator &);

struct F3 {
  template <typename T1, typename T2, typename T3>
  float operator()(T1 const &lhs, T2 const &chs, T3 const &rhs) const {
    PANIC();
  }
};

template GenericTensorAccessorW
    map_tensor_accessors3(GenericTensorAccessorR const &,
                          GenericTensorAccessorR const &,
                          GenericTensorAccessorR const &,
                          DataType,
                          F3 &&,
                          Allocator &);

} // namespace FlexFlow
