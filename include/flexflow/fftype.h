#ifndef _FF_TYPE_H
#define _FF_TYPE_H

#include "flexflow/ffconst.h"
#include <cstddef>

namespace FlexFlow {

class LayerID {
public:
  LayerID();
  LayerID(size_t id);
  bool is_valid_id() const;
  friend bool operator==(LayerID const &lhs, LayerID const &rhs);

public:
  size_t id;
};

}; // namespace FlexFlow

#endif // _FF_TYPE_H