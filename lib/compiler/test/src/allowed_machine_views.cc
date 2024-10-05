#include "compiler/allowed_machine_views.h"
#include "doctest/doctest.h"
#include "utils/containers/extend.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/fmt/unordered_set.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_allowed_machine_views") {

    SUBCASE("1 degree of parallelism") {
      MachineSpecification ms = MachineSpecification{
          /*num_nodes=*/1,
          /*num_cpus_per_node=*/5,
          /*num_gpus_per_node=*/5,
          /*inter_node_bandwidth=*/0,
          /*intra_node_bandwidth=*/0,
      };

      OperatorTaskSpace task = OperatorTaskSpace{{3}};

      std::unordered_set<MachineView> correct = {
          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0, /*device_idx=*/0, DeviceType::GPU},
              {MachineViewDimension{stride_t{1},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },

          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0, /*device_idx=*/1, DeviceType::GPU},
              {MachineViewDimension{stride_t{1},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },
          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0, /*device_idx=*/2, DeviceType::GPU},
              {MachineViewDimension{stride_t{1},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },
          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0, /*device_idx=*/0, DeviceType::GPU},
              {MachineViewDimension{stride_t{2},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineSpecification ms = MachineSpecification{3, 3, 3, 0, 0};
      OperatorTaskSpace task = OperatorTaskSpace{{2, 3}};

      auto make_2d_view = [&](int start_x,
                              int start_y,
                              int stride1,
                              int stride2,
                              MachineSpecificationDimension m1,
                              MachineSpecificationDimension m2) {
        return MachineView{
            MachineSpaceCoordinate{start_x, start_y, DeviceType::GPU},
            {MachineViewDimension{stride_t{stride1}, m1},
             MachineViewDimension{stride_t{stride2}, m2}},
        };
      };

      auto intra = MachineSpecificationDimension::INTRA_NODE;
      auto inter = MachineSpecificationDimension::INTER_NODE;
      std::unordered_set<MachineView> correct = {
          make_2d_view(0, 0, /*stride1=*/1, /*stride2=*/1, inter, intra),
          make_2d_view(1, 0, /*stride1=*/1, /*stride2=*/1, inter, intra),
          make_2d_view(0, 0, /*stride1=*/2, /*stride2=*/1, inter, intra),

          make_2d_view(0, 0, /*stride1=*/1, /*stride2=*/1, intra, inter),
          make_2d_view(0, 1, /*stride1=*/1, /*stride2=*/1, intra, inter),
          make_2d_view(0, 0, /*stride1=*/2, /*stride2=*/1, intra, inter),
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }
  }
}
