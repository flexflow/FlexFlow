#include "test/utils/all.h"
#include "test/utils/doctest.h"
#include "utils/containers.h"
#include "graph/labelled/output_labelled.h"

#include <string>
#include <vector>

using namespace FlexFlow ;

TEST_CASE("OutputLabelledMultiDiGraph implementation") {
    OutputLabelledMultiDiGraph g = OutputLabelledMultiDiGraph<std::string,std::string>::create<AdjacencyMultiDiGraph, UnorderedLabelling<Node,std::string>,UnorderedLabelling<MultiDiOutput, std::string>>();

    int num_nodes = 3;
    std::vector<std::string> nodel_labels =
      repeat(num_nodes, [&](int i) { return "nodel_labels_" + std::to_string(i); });
    
    std::vector<std::string> output_edge_labels = repeat(num_nodes, [&](int i) { return "output_edge_labels_" + std::to_string(i); });

    std::vector<NodePort> node_ports = repeat(num_nodes, [&] { return g.add_node_port(); });

    std::vector<Node> nodes = repeat(num_nodes, [&](NodePort p) {return g.add_node(p);});

    std::vector<NodePort> get_nodeports = repeat(num_nodes, [&](Node n) {return g.at(n);});

    CHECK(get_nodeports == node_ports);

    std::vector<std::string> output_labels = repeat(num_nodes, [&](int i) { return "output_labels_" + std::to_string(i); });

    //we should add_label for input and output 
    //(no,po,n1, p1), (n1,p1, n2, p2) , (n1,p1, n3, p3) this may have some problem, we can fix
    std::vector<MultiDiEdge> multi_diedges   = {{nodes[0], node_ports[0], nodes[1], node_ports[1]},
                                                  {nodes[1], node_ports[1], nodes[2], node_ports[2]},
                                                  {nodess[1], node_ports[1], nodes[3], nodde_ports[3]}};

    for(MultiDiEdge const & edge : multi_diedges) {
        OpenMultiDiEdge e{edge};
        g.add_edge(e);
    }

    std::vector<MultiDiOutput> multi_di_output = {{nodes[0], node_ports[0]},
                                                  {nodes[1], node_ports[1]},
                                                  {nodess[1], node_ports[1]}};
    
    for(int i = 0; i < output_labels.size(); i++) {
        g.add_output(multi_di_output[i], output_labels[i]);
    }

    std::vector<std::string> expected_output_labels;
    for(int i = 0; i < output_labels.size(); i++) {
        expected_output_labels.push_back(g.at(multi_di_output[i]));
    }

    CHECK(output_labels == expected_output_labels) ;

    CHECK(g.query_nodes(NodeQuery::all()) == without_order(nodes));

    CHECK(g.query_edges(OpenMultiDiEdgeQuery(MultiDiEdgeQuery::all())) ==without_order(multi_diedges)) ; //this may have some problem
    //add test for MultiDiEdgeQuery::with_src_nodes/with_dst_nodes/ with_src_idxs/with_dst_idxs
}