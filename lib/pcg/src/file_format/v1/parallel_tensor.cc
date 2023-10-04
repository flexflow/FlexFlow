#include "pcg/file_format/v1/parallel_tensor.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ParallelDim to_v1(ParallelDim const &dim) {
  return {dim.size, dim.degree, dim.is_replica_dim};
}

V1ParallelTensorShape to_v1(ParallelTensorShape const &shp) {
  std::vector<V1ParallelDim> pdims;
  for (ParallelDim const &pdim : shp.dims) {
    pdims.emplace_back(to_v1(pdim));
  }
  return {pdims, to_v1(shp.data_type)};
}

V1ParallelTensor to_v1(ParallelTensor const &t) {
  return {to_v1(t.get_shape()),
          to_v1(t.create_gradients),
          to_v1<V1Initializer>(t.initializer),
          to_v1<V1ParamSync>(t.sync_type),
          t.name};
}

} // namespace FlexFlow
