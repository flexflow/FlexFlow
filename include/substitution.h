/* Copyright 2021 CMU, Facebook, LANL, MIT, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _SUBSTITUTION_H_
#define _SUBSTITUTION_H_
#include "ffconst.h"
#include "tensor.h"
#include "graph.h"
#include <queue>
#include "tl/optional.h"

enum Compare {
  COMPARE_EQ,
  COMPARE_NE,
  COMPARE_LT,
  COMPARE_LE,
  COMPARE_GT,
  COMPARE_GE,
};

struct PMConstraint {
  PMConstraint(Compare comp, PMParameter para, int value);
  Compare comp;
  PMParameter para;
  int value;
};

struct TNConstraint {
  TNConstraint(Compare comp, TNParameter para, DIMParameter dim, int value);
  TNConstraint(Compare comp, TNParameter para1, DIMParameter dim1,
               TNParameter para2, DIMParameter dim2);
  bool singlePara;
  Compare comp;
  TNParameter para1, para2;
  DIMParameter dim1, dim2;
  int value;
};

class Op;
class OpX;
class GraphXfer;

struct TensorX {
  static const TensorX NO_TX;
  TensorX(void): op(NULL), idx(0) {}
  TensorX(OpX* _op, int _idx): op(_op), idx(_idx) {}
  Tensor to_tensor(const GraphXfer* xfer) const;
  OpX* op;
  int idx;
};

struct TensorXCompare {
  bool operator()(const TensorX& a, const TensorX& b) const {
    if (a.op != b.op) return a.op < b.op;
    return a.idx < b.idx;
  };
};

class OpX {
public:
  OpX(OperatorType type,
      int numInputs,
      int numOutputs,
      const TensorX& input1 = TensorX::NO_TX,
      const TensorX& input2 = TensorX::NO_TX,
      const TensorX& input3 = TensorX::NO_TX,
      const TensorX& input4 = TensorX::NO_TX);
  OpX(OperatorType type,
      int num_inputs,
      int num_outputs,
      const TensorX* inputs);
  bool add_pm_constraint(Compare, PMParameter para, int value);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, int);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, TNParameter, DIMParameter);
  bool get_pm_constraint(PMParameter para, int &value) const;
public:
  OperatorType type;
  Node mapOp;
  const OpX* matchOpX;
  std::vector<TensorX> inputs, weights, outputs;
  std::vector<PMConstraint> pmConstraints;
  std::vector<TNConstraint> tnConstraints;
};

template <bool include_input_xfer_cost>
class GraphCompare {
public:
  bool operator() (Graph* lhs, Graph* rhs) {
    return lhs->optimal_cost(include_input_xfer_cost) > rhs->optimal_cost(include_input_xfer_cost);
  }
};

class GraphXfer {
public:
  GraphXfer(FFModel* _model);
  TensorX new_tensor(void);
  bool can_match(OpX* srcOp, const Node& op, Graph* graph);
  void match(OpX* srcOp, const Node& op, Graph* graph);
  void unmatch(OpX* srcOp, const Node& op, Graph* graph);
  // Compute Ops
  template <typename T> 
  OpX* create_opx(const TensorX& input, const OpX* matchOpX);

  OpX* create_noop(const TensorX& input);
  OpX* create_concat(const TensorX* inputs,
                     int num_inputs,
                     const OpX* match_opx,
                     int concat_dim);
  OpX* create_element_binary(const TensorX& input1,
                             const TensorX& input2,
                             OperatorType op_type);
  OpX* create_element_unary(const TensorX& input,
                            OperatorType op_type);
  OpX* create_linear(const TensorX& input,
                     const OpX* match_opx,
                     int num_dims,
                     ActiMode acti_mode,
                     bool use_bias);
  OpX* create_conv2d(const TensorX& input,
                     const OpX* match_opx);
  OpX* create_attention(const TensorX& query,
                        const TensorX& key,
                        const TensorX& value,
                        const OpX* match_opx,
                        int num_heads);
  OpX* create_softmax(const TensorX& input,
                      int softmax_dim);
  // Parallel Ops
  OpX* create_repartition(const TensorX& input,
                          int repartition_dim,
                          int num_parts);
  OpX* create_replicate(const TensorX& input,
                        int replicate_dim,
                        int num_parts);
  OpX* create_reduction(const TensorX& input,
                        int reduction_dim,
                        int num_parts);
  OpX* create_combine(const TensorX& input,
                      int combine_dim,
                      int num_parts);
  bool map_output(const TensorX& src,
                  const TensorX& dst);

  
  Graph* create_new_graph(Graph* graph);
  bool create_new_operator(const OpX* opx, Node& op);

  std::string get_name() const;

  template <bool include_input_xfer_cost>
  void run(int depth, Graph* graph,
           std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare<include_input_xfer_cost>>&,
           std::unordered_set<size_t>&, float threshold, int maxNumOps, 
           int& num_matches_found, int& num_matches_rejected);
public:
  FFModel* model;
  tl::optional<std::string> name = tl::nullopt;
  int tensorId;
  std::map<Node, OpX*, NodeCompare> mappedOps;
  std::multimap<int, std::pair<Node, int> > mappedInputs;
  std::map<TensorX, TensorX, TensorXCompare> mappedOutputs;
  std::vector<OpX*> srcOps;
  std::vector<OpX*> dstOps;
};

class GraphSearchHelper {
public:
  GraphSearchHelper(FFModel *model);
  void graph_optimize(size_t budget, 
                      bool only_data_parallel,
                      Graph*& best_graph,
                      std::unordered_map<Node, MachineView>& optimal_views);
private:
  float sequence_optimize(Graph const *graph, 
                          Node const &sink_node, 
                          tl::optional<TensorShape> const &output_shape, 
                          tl::optional<TensorShape> const &input_shape);
  void load_graph_substitutions(std::vector<GraphXfer*> &xfers) const;
  Graph *construct_graph();
  void subgraph_optimize(Graph *subgraph);

  std::unique_ptr<Graph> base_optimize(Graph const *, bool include_input_xfer_cost);

  template <bool include_input_xfer_cost>
  std::unique_ptr<Graph> base_optimize(Graph const *);

  std::vector<TensorShape> possible_output_tensor_shapes(Node const &);
private:
  std::unordered_map<size_t, std::vector<std::unique_ptr<Graph>>> cached_optimized_graphs;

  FFModel* model;
  FFConfig const &config;
};

#endif
