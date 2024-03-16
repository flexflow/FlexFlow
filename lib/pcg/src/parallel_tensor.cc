#include "pcg/parallel_tensor.h"

namespace FlexFlow {

ParallelTensor::ParallelTensor(ParallelTensorDims const &dims,
                               DataType data_type,
                               CreateGrad create_gradients,
                               std::optional<ParamSync> sync_type,
                               std::optional<Initializer> initializer)
    : dims(dims), data_type(data_type), sync_type(sync_type),
      initializer(initializer), create_gradients(create_gradients) {}

} // namespace FlexFlow
