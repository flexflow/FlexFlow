#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_DIM_H

#include "utils/visitable.h"

namespace FlexFlow {

struct ParallelDim : public use_visitable_cmp<ParallelDim> {
public:
  ParallelDim() = delete;
  ParallelDim(size_t size, int degree, bool is_replica_dim = false);

public:
  size_t size;
  int degree;
  bool is_replica_dim;
};

bool is_valid(ParallelDim const &);
bool is_replica_dim(ParallelDim const &);

}

VISITABLE_STRUCT(::FlexFlow::ParallelDim, size, degree, is_replica_dim);
MAKE_VISIT_HASHABLE(::FlexFlow::ParallelDim);

namespace FlexFlow {

static_assert(is_equal_comparable<ParallelDim>::value, "ParallelDim must support ==");
static_assert(is_neq_comparable<ParallelDim>::value, "ParallelDim must support !=");
static_assert(is_lt_comparable<ParallelDim>::value, "ParallelDim must support <");
static_assert(!is_default_constructible<ParallelDim>::value, "ParallelDim must not be default constructible");
static_assert(is_copy_constructible<ParallelDim>::value, "ParallelDim must be copy constructible");

}

#endif
