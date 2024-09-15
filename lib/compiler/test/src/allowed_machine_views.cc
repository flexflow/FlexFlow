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

      MachineSpecification ms = MachineSpecification{1, 5, 5, 0, 0};
      TaskSpaceOperator task = TaskSpaceOperator{{num_points_t{3}}};

      std::unordered_set<MachineView> correct = {
          MachineView{{{stride_t{1}}},
                      {{MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{0, 0, DeviceType::GPU}},

          MachineView{{{stride_t{1}}},
                      {{MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{0, 1, DeviceType::GPU}},
          MachineView{{{stride_t{1}}},
                      {{MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{0, 2, DeviceType::GPU}},
          MachineView{{{stride_t{2}}},
                      {{MachineSpecificationDimension::INTRA_NODE}},
                      MachineSpaceCoordinate{0, 0, DeviceType::GPU}},
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineSpecification ms = MachineSpecification{3, 3, 3, 0, 0};
      TaskSpaceOperator task =
          TaskSpaceOperator{{num_points_t{2}, num_points_t{3}}};

      auto make_2d_views = [&](int start_x,
                               int start_y,
                               int stride1,
                               int stride2,
                               MachineSpecificationDimension m1,
                               MachineSpecificationDimension m2) {
        return MachineView{
            {stride_t{stride1}, stride_t{stride2}},
            {m1, m2},
            MachineSpaceCoordinate{start_x, start_y, DeviceType::GPU}};
      };

      auto intra = MachineSpecificationDimension::INTRA_NODE;
      auto inter = MachineSpecificationDimension::INTER_NODE;
      std::unordered_set<MachineView> correct = {
          make_2d_views(0, 0, /*stride1*/ 1, /*stride2*/ 1, inter, intra),
          make_2d_views(1, 0, /*stride1*/ 1, /*stride2*/ 1, inter, intra),
          make_2d_views(0, 0, /*stride1*/ 2, /*stride2*/ 1, inter, intra),

          make_2d_views(0, 0, /*stride1*/ 1, /*stride2*/ 1, intra, inter),
          make_2d_views(0, 1, /*stride1*/ 1, /*stride2*/ 1, intra, inter),
          make_2d_views(0, 0, /*stride1*/ 2, /*stride2*/ 1, intra, inter),
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }
  }
}
