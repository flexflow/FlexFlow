#ifndef _FF_TYPE_H
#define _FF_TYPE_H

#include "flexflow/ffconst.h"
#include <cstddef>
#include <functional>

namespace FlexFlow {

class LayerID {
public:
  static const LayerID NO_ID;
  LayerID();
  LayerID(size_t id, size_t transformer_layer_id);
  bool is_valid_id() const;
  friend bool operator==(LayerID const &lhs, LayerID const &rhs);

public:
  size_t id, transformer_layer_id;
};

class PEFTModelID {
public:
  static const PEFTModelID NO_ID;
  PEFTModelID();
  PEFTModelID(size_t id);
  bool is_valid_id() const;
  friend bool operator==(PEFTModelID const &lhs, PEFTModelID const &rhs);

public:
  size_t id;
};

}; // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::PEFTModelID> {
  size_t operator()(FlexFlow::PEFTModelID const &n) const {
    return n.id;
  }
};
} // namespace std

#endif // _FF_TYPE_H
