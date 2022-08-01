#ifndef _NODE_H
#define _NODE_H

#include "tl/optional.h"

namespace FlexFlow {

class Op;

namespace PCG {

struct Node {
  Node(void);
  Node(size_t _guid, Op *_ptr) : guid(_guid), ptr(_ptr) {}
  inline bool operator==(const Node &b) const {
    if (guid != b.guid)
      return false;
    if (ptr != b.ptr)
      return false;
    if (original_guid != b.original_guid)
      return false;
    return true;
  }
  inline bool operator!=(const Node &b) const {
    if (guid != b.guid)
      return true;
    if (ptr != b.ptr)
      return true;
    if (original_guid != b.original_guid)
      return false;
    return false;
  }
  inline bool operator<(const Node &b) const {
    if (guid != b.guid)
      return guid < b.guid;
    if (ptr != b.ptr)
      return ptr < b.ptr;
    if (original_guid != b.original_guid)
      return false;
    return false;
  }
  Node &operator=(const Node &n) {
    guid = n.guid;
    ptr = n.ptr;
    original_guid = n.original_guid;
    return *this;
  }
  std::string op_to_string(const Op *ptr) const;
  std::string to_string(void) const {
    if (ptr != NULL) {
      return op_to_string(ptr) + "_" + std::to_string(guid);
    } else {
      return "UnmappedOp_" + std::to_string(guid);
    }
  }
  static const Node INVALID_NODE;
  size_t guid;
  const Op *ptr;

  tl::optional<size_t> original_guid = tl::nullopt;
};

}; // namespace PCG

}; // namespace FlexFlow

#endif // _NODE_H