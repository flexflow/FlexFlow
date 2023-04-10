#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_USES_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_USES_H

#include "parallel_tensor.h"
#include "tensor_uses.h"

namespace FlexFlow {

/* struct ParallelTensorUseDescription { */
/*   ParallelTensorUseDescription() = delete; */
/*   ParallelTensorUseDescription(TensorUseType const &, Op const *, int); */

/*   TensorUseType type; */ 
/*   Op const *op; */
/*   int idx; */
/* }; */

struct ParallelTensorSourceInfo {
  ParallelTensorSourceInfo() = delete;
  ParallelTensorSourceInfo(Op const *, int);

  Op const *op;
  int idx;
};

struct ParallelTensorUses {
  ParallelTensorUses() = default;

  /* std::vector<ParallelTensorUseDescription> at(ParallelTensor const &) const; */
  /* std::vector<ParallelTensorUseDescription> at(ParallelTensorBase const *) const; */
  /* std::vector<ParallelTensorUseDescription> at(size_t parallel_tensor_guid) const; */

  optional<ParallelTensorSourceInfo> get_source(ParallelTensor const &) const;

  void update(Op const &);
  void remove(Op const &);
private:
  std::unordered_map<size_t, size_t> uses;
  /* std::unordered_map<size_t, std::vector<ParallelTensorUseDescription>> uses; */
};

}

#endif
