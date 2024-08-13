#include "compiler/machine_mapping.h"
#include "doctest/doctest.h"
#include "test_generator.h"

TEST_SUITE(FF_TEST_SUITE) {

  // TEST_CASE("MachineMapping::combine") {
  //   RC_SUBCASE([](MachineMapping const &m0, MachineMapping const &m1) {
  //     RC_PRE(MachineMapping::nodes_are_disjoint(m0, m1));

  //     MachineMapping comb = MachineMapping::combine(m0, m1);

  //     RC_ASSERT(comb.machine_views.size() ==
  //               m0.machine_views.size() + m1.machine_views.size());
  //     RC_ASSERT(is_submap(comb.machine_views, m0.machine_views));
  //     RC_ASSERT(is_submap(comb.machine_views, m1.machine_views));
  //   });
  // }

  // TEST_CASE("OptimalCostResult::infinity") {
  //   RC_SUBCASE([](OptimalCostResult const &c) {
  //     RC_ASSERT(c.runtime <= OptimalCostResult::infinity().runtime);
  //   });
  // }
}
