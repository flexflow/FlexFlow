#include "op-attrs/parallel_dim.h"

namespace FlexFlow {

ParallelDim::ParallelDim(size_t size, int degree, bool is_replica_dim) 
  : size(size), degree(degree), is_replica_dim(is_replica_dim)
{ }


bool is_valid(ParallelDim const &dim) {
  return dim.size > 0 && dim.degree >= 1 && dim.size % dim.degree == 0;
}

}
