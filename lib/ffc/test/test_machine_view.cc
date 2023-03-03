#include "flexflow/config.h"
#include "flexflow/machine_view.h"
#include "gtest/gtest.h"

using namespace Legion;
using namespace FlexFlow;

TEST(machine_view_get_domain, basic) {
  MachineView mv;
  mv.ndims = 1;
  mv.start_device_id = 2;
  mv.dim[0] = 2;
  mv.stride[0] = 1;

  Domain d;
  d.dim = 1;
  d.rect_data[0] = 0;
  d.rect_data[0 + d.dim] =
      1; // Domain is includes, MachineView is exclusive on hi

  EXPECT_EQ(mv.get_domain(), d);
}

TEST(machine_view_get_device_id, basic) {
  MachineView mv;
  mv.ndims = 1;
  mv.start_device_id = 2;
  mv.dim[0] = 2;
  mv.stride[0] = 1;

  EXPECT_EQ(mv.get_device_id({0}), 2);
  EXPECT_EQ(mv.get_device_id({1}), 3);
}
