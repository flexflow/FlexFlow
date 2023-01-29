#ifndef _FLEXFLOW_FFC_SIMPLIFICATION_H
#define _FLEXFLOW_FFC_SIMPLIFICATION_H

namespace FlexFlow {
namespace PCG {

struct SimplificationSettings {
  bool simplify_parallel_ops = false;
  bool fuse_parallel_ops = false;
  bool remove_trailing_parallel_ops = false;
  bool remove_noops = false;
};

}
}

#endif 
