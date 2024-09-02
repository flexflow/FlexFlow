#include "compiler/machine_mapping/machine_mapping.h"
#include "cost_estimator_for_test.h"
#include "doctest/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("combine") {
    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));
    MachineMapping machine_mapping_0({{Node(0), machine_view_0}});
    MachineMapping machine_mapping_1({{Node(1), machine_view_1}});
    MachineMapping combined({{Node(0), machine_view_0}, {Node(1), machine_view_1}});
    MachineMapping result = combine(machine_mapping_0, machine_mapping_1);
    CHECK(result == combined);
  }

  TEST_CASE("nodes_are_disjoint") {
    MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));
    MachineMapping machine_mapping_0({{Node(0), machine_view_0}});
    MachineMapping machine_mapping_1({{Node(1), machine_view_1}});
    MachineMapping combined({{Node(0), machine_view_0}, {Node(1), machine_view_1}});
    CHECK(nodes_are_disjoint(machine_mapping_0, machine_mapping_1));
    CHECK_FALSE(nodes_are_disjoint(machine_mapping_0, combined));
  }
}
