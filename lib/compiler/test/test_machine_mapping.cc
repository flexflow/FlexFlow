#include "doctest.h"
#include "test_generator.h"

TEST_CASE("MachineMapping::sequential_combine") {
  rc::check([](MachineMapping const &mp0, MachineMapping const &mp1) {
    RC_PRE(are_disjoint(keys(mp0), keys(mp1)));

    MachineMapping comb = MachineMapping::sequential_combine(mp0, mp1);

    RC_ASSERT(comb.runtime == mp0.runtime + mp1.runtime);
    RC_ASSERT(comb.machine_views.size() ==
              mp0.machine_views.size() + mp1.machine_views.size());
    for (auto p : mp0.machine_views) {
      RC_ASSERT(p.second == comb.machine_views.at(p.first));
    }
  });
}

TEST_CASE("MachineMapping::parallel_combine") {
  rc::check([](MachineMapping const &mp0, MachineMapping const &mp1) {
    RC_PRE(are_disjoint(keys(mp0), keys(mp1)));

    MachineMapping comb = MachineMapping::parallel_combine(mp0, mp1);

    RC_ASSERT(comb.runtime == std::max(mp0.runtime, mp1.runtime));
    RC_ASSERT(comb.machine_views.size() ==
              mp0.machine_views.size() + mp1.machine_views.size());
    for (auto p : mp0.machine_views) {
      RC_ASSERT(p.second == comb.machine_views.at(p.first));
    }
  });
}

TEST_CASE("MachieMapping::infinity") {
  rc::check([](MachineMapping const &mp) {
    RC_ASSERT(mp.runtime <= MachineMapping::infinity().runtime);
  });
}