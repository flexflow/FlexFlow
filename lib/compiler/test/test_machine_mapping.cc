#include "doctest.h"
#include "test_generator.h"

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

TEST_CASE("MachineMapping::sequential_combine") {
  rc::check([](MachineMapping const &mp0, MachineMapping const &mp1) {
    RC_PRE(nodes_are_disjoint(mp0, mp1));

    MachineMapping comb = MachineMapping::sequential_combine(mp0, mp1);

    RC_ASSERT(comb.runtime == mp0.runtime + mp1.runtime);
    RC_ASSERT(comb.machine_views.size() ==
              mp0.machine_views.size() + mp1.machine_views.size());
    RC_ASSERT(is_submap(comb.machine_views, mp0.machine_views));
    RC_ASSERT(is_submap(comb.machine_views, mp1.machine_views));
  });
}

TEST_CASE("MachineMapping::parallel_combine") {
  rc::check([](MachineMapping const &mp0, MachineMapping const &mp1) {
    RC_PRE(nodes_are_disjoint(mp0, mp1));

    MachineMapping comb = MachineMapping::parallel_combine(mp0, mp1);

    RC_ASSERT(comb.runtime == std::max(mp0.runtime, mp1.runtime));
    RC_ASSERT(comb.machine_views.size() ==
              mp0.machine_views.size() + mp1.machine_views.size());
    RC_ASSERT(is_submap(comb.machine_views, mp0.machine_views));
    RC_ASSERT(is_submap(comb.machine_views, mp1.machine_views));
  });
}

TEST_CASE("MachineMapping::infinity") {
  rc::check([](MachineMapping const &mp) {
    RC_ASSERT(mp.runtime <= MachineMapping::infinity().runtime);
  });
}