# substitutions

## Substitution

A substitution is to replace a subgraph of the PCG by a new one. We refer to the subgraph to be replaced as the input graph, and the new subgraph to replace the input graph as the output graph.

A `Substitution` object describes a substitution. It consists of
* An `input_graph` of type `GraphPattern` that describes which kind of input graphs the substitution can be applied to;
* An `output_graph` of type `OutputGraphExpr` that describes how the output graph is computed from the input graph; and
* An `input_mapping` and `output_maping` that describes how the output graph is connected to the original PCG.

### GraphPattern and MultiDiGraphPatternMatch

A `GraphPattern` is defined as an open graph with node label `OperatorPattern` and output label `ParallelTensorPattern`, which is refered to as the pattern graph. The graph structure of a `GraphPattern` instance defines the geometrical property of the input graph, while the node labels and output labels define the attribute property of that.

To apply a substitution to a PCG, we should first match the pattern graph to a subgraph of the PCG. `MultiDiGraphPatternMatch` describes the match, which consists of
* `node_assignment`: a mapping from the nodes of the pattern graph to the nodes of the PCG; and
* `edge_assignment`: a mapping from the edges of the pattern graph to the nodes of the PCG.
The input graph derived by this match is then defined by `values(node_assignment)` and `values(edge_assignment)`. A match is valid if and only if
* `node_assignment` and `edge_assignment` are injections;
* For every node `n` in the pattern graph, `edge_assignment` derives a bijection between `query_edges({n})` and `query_edges({node_assignment.at_l(n)})`.

### OutputGraphExpr

An `OutputGraphExpr` is defined as an open graph with node label `OperatorAttrAssignment` and output label `ParallelTensorAttrAssignment`, which defines how the operator attributes and the parallel tensor attributes of the output graph are derived from the input graph.

`OperatorAttrAssignment` is a collection of `OperatorAttributeKey` and `GraphAttributeExpr` pairs. It defines how the attributes of a single operator is calculated from the input graph. A pair `{operator_attribute_key, graph_attribute_expr}` in the collection means the value of `graph_attribute_expr` is assigned to the attribute named `operator_attribute_key` of the operator.

`ParallelTensorAttrAssignment` is defined in the similar way to `OperatorAttrAssignment`.

`GraphAttributeExpr` is defined as one of `NodeAttrAccess`, `EdgeAttrAccess` and `AttrConstant`:
* `NodeAttrAccess` consists of a node `node` and an expression `attr_expr` on the attributes of the operator associated with the node. The value of a `NodeAttrAccess` instance is the value of `attr_expr` evaluated on the operator associated with the node.
* `EdgeAttrAccess` is defined in the similar way to `NodeAttrAccess`.
* `AttrConstant` consists of a constant `value`. The value of an `AttrConstant` instance is `value`.
