#ifndef _FF_TYPE_H
#define _FF_TYPE_H

#include "flexflow/ffconst.h"
#include <cstddef>

namespace FlexFlow {

class LayerID {
public:
  static const LayerID NO_ID;
  LayerID();
  LayerID(size_t id, size_t transformer_layer_id, size_t model_id);
  bool is_valid_id() const;
  friend bool operator==(LayerID const &lhs, LayerID const &rhs);

public:
  size_t id, transformer_layer_id, model_id;
};

}; // namespace FlexFlow

#endif // _FF_TYPE_H
