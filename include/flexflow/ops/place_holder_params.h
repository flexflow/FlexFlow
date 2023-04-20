#ifndef _FLEXFLOW_PLACE_HOLDER_PARAMS_H
#define _FLEXFLOW_PLACE_HOLDER_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct PlaceHolderParams {
  LayerID layer_guid;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(PlaceHolderParams const &, PlaceHolderParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::PlaceHolderParams> {
  size_t operator()(FlexFlow::PlaceHolderParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_PLACE_HOLDER_PARAMS_H
