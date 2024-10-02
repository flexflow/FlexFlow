#include "compiler/machine_mapping/machine_mapping_result.h"
#include "cost_estimator_for_test.h"
#include "doctest/doctest.h"
#include "pcg/machine_view.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("sequential_combine") {
    FAIL("TODO");
    // MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    // MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));
    // parallel_layer_guid_t layer0 = parallel_layer_guid_t(Node(0));
    // parallel_layer_guid_t layer1 = parallel_layer_guid_t(Node(1));
    // MachineMapping machine_mapping_empty(
    //     std::unordered_map<parallel_layer_guid_t, MachineView>{});
    // MachineMapping machine_mapping_0({{layer0, machine_view_0}});
    // MachineMapping machine_mapping_1({{layer1, machine_view_1}});
    // MachineMapping combined(
    //     {{layer0, machine_view_0}, {layer1, machine_view_1}});
    // MachineMappingResult s0(0, machine_mapping_empty);
    // MachineMappingResult s1(1, machine_mapping_0);
    // MachineMappingResult s2(2, machine_mapping_1);
    //
    // float comm_cost = 2.0;
    //
    // MachineMappingResult result0 = sequential_combine(s0, comm_cost, s1);
    // CHECK(result0.runtime == 1);
    // CHECK(result0.machine_mapping == machine_mapping_0);
    //
    // MachineMappingResult result1 = sequential_combine(s0, comm_cost, s2);
    // CHECK(result1.runtime == 2);
    // CHECK(result1.machine_mapping == machine_mapping_1);
    //
    // MachineMappingResult result2 = sequential_combine(s1, comm_cost, s2);
    // CHECK(result2.runtime == 3);
    // CHECK(result2.machine_mapping == combined);
  }

  TEST_CASE("parallel_combine") {
    FAIL("TODO");
    // MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    // MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));
    // parallel_layer_guid_t layer0 = parallel_layer_guid_t(Node(0));
    // parallel_layer_guid_t layer1 = parallel_layer_guid_t(Node(1));
    // MachineMapping machine_mapping_empty(
    //     std::unordered_map<parallel_layer_guid_t, MachineView>{});
    // MachineMapping machine_mapping_0({{layer0, machine_view_0}});
    // MachineMapping machine_mapping_1({{layer1, machine_view_1}});
    // MachineMapping combined(
    //     {{layer0, machine_view_0}, {layer1, machine_view_1}});
    // MachineMappingResult s0(0, machine_mapping_empty);
    // MachineMappingResult s1(1, machine_mapping_0);
    // MachineMappingResult s2(2, machine_mapping_1);
    //
    // MachineMappingResult result0 = parallel_combine(s0, s1);
    // CHECK(result0.runtime == 1);
    // CHECK(result0.machine_mapping == machine_mapping_0);
    //
    // MachineMappingResult result1 = parallel_combine(s0, s2);
    // CHECK(result1.runtime == 2);
    // CHECK(result1.machine_mapping == machine_mapping_1);
    //
    // MachineMappingResult result2 = parallel_combine(s1, s2);
    // CHECK(result2.runtime == 2);
    // CHECK(result2.machine_mapping == combined);
  }

  TEST_CASE("minimize_runtime") {
    FAIL("TODO");
    // MachineView machine_view_0 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(1));
    // MachineView machine_view_1 = make_1d_machine_view(gpu_id_t(0), gpu_id_t(2));
    // parallel_layer_guid_t layer0 = parallel_layer_guid_t(Node(0));
    // parallel_layer_guid_t layer1 = parallel_layer_guid_t(Node(1));
    // MachineMapping machine_mapping_empty(
    //     std::unordered_map<parallel_layer_guid_t, MachineView>{});
    // MachineMapping machine_mapping_0({{layer0, machine_view_0}});
    // MachineMapping machine_mapping_1({{layer1, machine_view_1}});
    // MachineMapping combined(
    //     {{layer0, machine_view_0}, {layer1, machine_view_1}});
    // MachineMappingResult s0(0, machine_mapping_empty);
    // MachineMappingResult s1(1, machine_mapping_0);
    // MachineMappingResult s2(2, machine_mapping_1);
    //
    // MachineMappingResult _s0 = s0;
    // MachineMappingResult _s1 = s1;
    // MachineMappingResult _s2 = s2;
    //
    // minimize_runtime(_s0, _s1);
    // CHECK(_s0 == s0);
    // minimize_runtime(_s1, _s2);
    // CHECK(_s1 == s1);
    //
    // minimize_runtime(_s1, _s0);
    // CHECK(_s1 == s0);
    //
    // minimize_runtime(_s2, get_infinity_machine_mapping_result());
    // CHECK(_s2 == s2);
  }
}
