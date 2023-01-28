#include "op-meta/ops/conv_2d_params.h"
#include "utils/hash-utils.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"

namespace FlexFlow {

namespace Input {
  constexpr int WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM = 5;
}

namespace Output {
  constexpr int WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3, REPLICA = 4, NUMDIM = 5;
}

namespace Kernel {
  constexpr int WIDTH = 0, HEIGHT = 1, CHANNEL_IN = 2, CHANNEL_OUT = 3, REPLICA = 4;
}

namespace Bias {
  constexpr int CHANNEL = 0, REPLICA_1 = 1, REPLICA_2 = 2, REPLICA_3 = 3, REPLICA_4 = 4;
}

typename Conv2DParams::AsConstTuple Conv2DParams::as_tuple() const {
  return { this->out_channels, this->kernel_h, this->kernel_w, this->stride_h, this->stride_w, this->padding_h, this->padding_w, this->groups, this->activation, this->use_bias };
};

bool operator==(Conv2DParams const &lhs, Conv2DParams const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(Conv2DParams const &lhs, Conv2DParams const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

}

namespace std {

}
