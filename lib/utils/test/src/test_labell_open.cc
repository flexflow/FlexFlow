#include "test/utils/all.h"
#include "test/utils/rapidcheck/visitable.h"
#include "utils/containers.h"
#include "utils/graph/labelled_open.h"

#include <string>

using namespace FlexFlow;

using namespace rc;

//test the LabelledOpenMultiDiGraph

TEST_CASE_TEMPLATE("LabelledOpenMultiDiGraph implementations", T,  UnorderedLabelledOpenMultiDiGraph) {
    // I define NodeLabel/ as int, EdgeLabelInputLabel/OutputLabel as string 
    LabelledOpenMultiDiGraph<int, std::string> g = LabelledOpenMultiDiGraph<int, std::string>::create<UnorderedLabelledOpenMultiDiGraph<int, std::string>>();
    int num_nodes = *gen::inRange(1, 10);
    std::vector<Node> n = repeat<Node>(num_nodes, [&g](int i) {
    return g.add_node(i);
});
    for(int i =0; i < num_nodes; i++) {
        CHECK( i == g.at(n[i]));// check NodeLabel &at(Node const &n);
    }
    NOT_IMPLEMENTED();
    
}