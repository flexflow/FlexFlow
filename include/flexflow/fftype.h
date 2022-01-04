#pragma once
#include "flexflow/ffconst.h"

namespace FlexFlow {

class LayerID {
public:
  LayerID() : id(0) {}
  LayerID(size_t _id) : id(_id) {
    assert(is_valid_id());
  }
  bool is_valid_id() const {
    return (id >= LAYER_GUID_FIRST_VALID && id <= LAYER_GUID_LAST_VALID);
  }
  bool operator == (const LayerID & other) const {
    return this->id == other.id;
  }
public:
  size_t id;
};

};
