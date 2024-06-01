
#include "kernels/attention_kernels.h"
#include "kernels/batch_norm_kernels.h"
#include "kernels/conv_2d_kernels.h"
#include "kernels/dropout_kernels.h"
#include "kernels/element_binary_kernels.h"
#include "kernels/element_unary_kernels.h"
#include "kernels/gather_kernels.h"
#include "kernels/layer_norm_kernels.h"
#include "kernels/linear_kernels.h"
#include "kernels/partition_kernels.h"
#include "kernels/pool_2d_kernels.h"
#include "kernels/reduce_kernels.h"
#include "kernels/reduction_kernels.h"
#include "kernels/reshape_kernels.h"
#include "kernels/softmax_kernels.h"
#include "kernels/topk_kernels.h"
#include "kernels/transpose_kernels.h"
#include <variant>

namespace FlexFlow {

using DeviceStates = std::variant<MHAPerDeviceState,
                                  BatchNormPerDeviceState,
                                  Conv2DPerDeviceState,
                                  DropoutPerDeviceState,
                                  ElementBinaryPerDeviceState,
                                  ElementUnaryPerDeviceState,
                                  GatherPerDeviceState,
                                  LayerNormPerDeviceState,
                                  LinearPerDeviceState,
                                  Pool2DPerDeviceState,
                                  ReducePerDeviceState,
                                  RepartitionPerDeviceState,
                                  ReshapePerDeviceState,
                                  SoftmaxPerDeviceState,
                                  TopKPerDeviceState,
                                  TransposePerDeviceState>;

}
