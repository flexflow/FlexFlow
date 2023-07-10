#include "op-attrs/ops/linear.h"

namespace FlexFlow {

LinearAttrs::LinearAttrs(int _out_channels,
                         bool _use_bias,
                         DataType _data_type,
                         Activation _activation)
    : out_channels(_out_channels), use_bias(_use_bias), data_type(_data_type),
      activation(_activation) {}

} // namespace FlexFlow
