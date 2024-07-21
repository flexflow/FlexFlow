#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/get_only.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("topological_ordering") {
    // TODO(@lockshaw) should probably be replaced with a rapidcheck test that
    // compares ParallelComputationGraph to DataflowGraph, but since we
    // currently don't have rapidcheck generation for DataflowGraph this will
    // have to do for now

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ParallelLayerAttrs layer_label = some<ParallelLayerAttrs>();
    ParallelTensorAttrs tensor_label = some<ParallelTensorAttrs>();

    ParallelLayerAddedResult layer1_added =
        add_parallel_layer(pcg, layer_label, {}, {tensor_label});
    parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
    parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

    ParallelLayerAddedResult layer2_added =
        add_parallel_layer(pcg, layer_label, {tensor1}, {tensor_label});
    parallel_layer_guid_t layer2 = layer2_added.parallel_layer;
    parallel_tensor_guid_t tensor2 = get_only(layer2_added.outputs);

    ParallelLayerAddedResult layer3_added =
        add_parallel_layer(pcg, layer_label, {tensor2}, {tensor_label});
    parallel_layer_guid_t layer3 = layer3_added.parallel_layer;
    parallel_tensor_guid_t tensor3 = get_only(layer3_added.outputs);

    std::vector<parallel_layer_guid_t> result = topological_ordering(pcg);
    // std::vector<parallel_layer_guid_t> correct = {layer1, layer2, layer3};
    // CHECK(result == correct);
  }
}
