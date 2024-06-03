## Tutorial of substitution lib with simple example

#### Create a pattern

```c++
//we should specify both the node pattern and edge pattern when defining a GraphPattern 

//first define an operator pattern for example, specify the node to have a linear 
//operator
OperatorPattern operator_pattern_n0{
        std::vector<OperatorAttributeConstraint>{OperatorAttributeConstraint{
            ConstraintType::EQUAL, OperatorAttributeKey::OP_TYPE, Op::LINEAR}}};

//then define a tensor_pattern that restrict the pattern of edge in pcg. for example, 
//specify that the first dimension (indexed by 0) of a tensor should be 2
ParallelTensorPattern tensor_pattern_e0{
    std::vector<TensorAttributeConstraint>{
        TensorAttributeConstraint{ConstraintType::EQUAL,
                                    ListIndexAccess<TensorAttributeKey>{
                                        TensorAttributeKey::DIM_SIZES, 0},
                                    2}}};
/*
remeber that both operator_pattern and tensor_pattern are std::vector, meaning that you 
can define more than one constraint depending on the context
*/
```


#### Pack into GraphPattern
```c++
//create a graph with node label of OperatorPattern and edge label of ParallelTensorPattern
auto ig =
    OutputLabelledOpenMultiDiGraph<OperatorPattern, ParallelTensorPattern>::
        create<UnorderedOutputLabelledOpenMultiDiGraph<
            OperatorPattern,
            ParallelTensorPattern>>();
//add constraints defined above as argument to create a node
Node n0 = ig.add_node(operator_pattern_n0);
//add port number to distinguish different edges going to the same node
NodePort p0 = ig.add_node_port();
//create edge
InputMultiDiEdge e0{n0, p0, std::make_pair(p0.value(), p0.value())};
ig.add_edge(e0);
//add edge constraints above to the edge e0
ig.add_label(e0, tensor_pattern_e0);

//a pattern graph with one input edge pointing to a node
/*
        n0 (Linear)
        ↑
*/
RC_ASSERT(get_nodes(ig).size() == 1);
RC_ASSERT(get_edges(ig).size() == 1);
```

#### Define OutputGraph
```cpp

//define a 3-node PCG that can be applied from the input graph ig

//Partition node that can partite the input into two parts
OperatorAttrAssignment op_ass_n1{
        {{OperatorAttributeKey::OP_TYPE, AttrConstant{Op::REPARTITION}},
         {OperatorAttributeKey::PARALLEL_DIM, AttrConstant{ff_dim_t{0}}},
         {OperatorAttributeKey::PARALLEL_DEGREE, AttrConstant{2}}}};

//Linear node
OperatorAttrAssignment op_ass_n2{
    {{OperatorAttributeKey::OP_TYPE, AttrConstant{Op::LINEAR}},
        {OperatorAttributeKey::OUT_CHANNELS,
        OperatorAttrAccess{n0, OperatorAttributeKey::OUT_CHANNELS}},
        {OperatorAttributeKey::USE_BIAS,
        OperatorAttrAccess{n0, OperatorAttributeKey::USE_BIAS}},
        {OperatorAttributeKey::DATA_TYPE,
        OperatorAttrAccess{n0, OperatorAttributeKey::DATA_TYPE}},
        {OperatorAttributeKey::ACTIVATION,
        OperatorAttrAccess{n0, OperatorAttributeKey::ACTIVATION}},
        {OperatorAttributeKey::REGULARIZER,
        OperatorAttrAccess{n0, OperatorAttributeKey::REGULARIZER}}}};

//Reduce node that will combine the result of two partitions
OperatorAttrAssignment op_ass_n3{
    {{OperatorAttributeKey::OP_TYPE, AttrConstant{Op::REDUCTION}},
    {OperatorAttributeKey::PARALLEL_DIM, AttrConstant{ff_dim_t{0}}},
    {OperatorAttributeKey::PARALLEL_DEGREE, AttrConstant{2}}}};

//notice that these assignments will be evaluated 
//into new operators in the apply_substitution function 
//and be inserted into the new pcg

//create outputgraph with 3 nodes and 3 edges
auto og = NodeLabelledOpenMultiDiGraph<OperatorAttrAssignment>::create<
    UnorderedNodeLabelledOpenMultiDiGraph<OperatorAttrAssignment>>();
Node n1 = og.add_node(op_ass_n1);
Node n2 = og.add_node(op_ass_n2);
Node n3 = og.add_node(op_ass_n3);
NodePort p1 = og.add_node_port();
NodePort p2 = og.add_node_port();
NodePort p3 = og.add_node_port();

InputMultiDiEdge e1{n1, p1, {p1.value(), p1.value()}};
MultiDiEdge e2{n2, p2, n1, p1};
MultiDiEdge e3{n3, p3, n2, p2};
og.add_edge(e1);
og.add_edge(e2);
og.add_edge(e3);
OutputGraphExpr output_graph_expr{og};

/*
The output graph looks like this
               n3 (Reduce)
               ↑
               n2 (Linear)
               ↑
               n1 (Partition)
               ↑
*/
RC_ASSERT(get_nodes(og).size() == 3);
RC_ASSERT(get_edges(og).size() == 3);
```

#### Define substitution
```cpp
//define two dict that specify how the input and output edges are mapped in the substitution
bidict<InputMultiDiEdge, InputMultiDiEdge> input_mapping;
input_mapping.equate(e0, e1);
bidict<OutputMultiDiEdge, OutputMultiDiEdge> output_mapping;

Substitution substitution{
    input_graph, output_graph_expr, input_mapping, output_mapping};
```

#### Apply substitution
```cpp

//create the target pcg that we want to apply for substitution
SubParallelComputationGraph pcg =
    OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::create<
        UnorderedOutputLabelledOpenMultiDiGraph<Operator,
                                                ParallelTensor>>();

Node n4 = pcg.add_node(Operator{InputAttrs{}, "input"});
Node n5 = pcg.add_node(Operator{
    LinearAttrs{1, false, DataType::FLOAT, Activation::RELU, std::nullopt},
    "linear"});
NodePort p4 = pcg.add_node_port();
NodePort p5 = pcg.add_node_port();

MultiDiEdge e4{n5, p5, n4, p4};
pcg.add_edge(e4);
pcg.add_label(e4,
                ParallelTensor(ParallelTensorDims({2, 1}),
                                DataType::FLOAT,
                                CreateGrad::YES));

/* Our target pcg looks like this
           n5 (Linear)
           ↑
           n4 (input)
*/

//create criterion function that will test every predefined edge and node constraints
MatchAdditionalCriterion criterion{
    [&](Node const &pattern_node, Node const &graph_node) {
        return operator_satisfies(pcg.at(graph_node),
                                input_graph.value().at(pattern_node));
    },
    [&](OpenMultiDiEdge const &pattern_edge,
        OpenMultiDiEdge const &graph_edge) {
        return parallel_tensor_satisfies(
            pcg.at(graph_edge), input_graph.value().at(pattern_edge));
    }};

RC_ASSERT(criterion.node_criterion(n0, n5));


//find the match point that we can apply the substitution in the target pcg
std::vector<MultiDiGraphPatternMatch> matches =
    find_pattern_matches(input_graph, pcg, criterion);

//there is only one match point in the pcg that we defined
RC_ASSERT(matches.size() == 1);

//apply substitution
//the number of new pcg generated is bounded by O(2^(sn))where s is the number of
//different substitutions and n is the number of nodes
SubParallelComputationGraph new_pcg =
    apply_substitution(pcg, substitution, matches[0]);

//now the new pcg becomes as follow
/*
    n3 (Reduce)
    ↑
    n2 (Linear)
    ↑
    n1 (Partition)
    ↑
    n4 (Input)
*/
RC_ASSERT(get_nodes(new_pcg).size() == 4);
RC_ASSERT(get_edges(new_pcg).size() == 3);
```




