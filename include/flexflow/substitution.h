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

#ifndef _FLEXFLOW_SUBSTITUTION_H_
#define _FLEXFLOW_SUBSTITUTION_H_
#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"
#include "flexflow/graph.h"
#include <queue>
#include "tl/optional.h"
#include "flexflow/utils/recursive_logger.h"
#include "flexflow/substitution_loader.h"

namespace FlexFlow::PCG {

namespace sl = FlexFlow::substitution_loader;

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
  tl::optional<ParallelTensor> to_tensor(const GraphXfer* xfer) const;
  OpX* op;
  int idx;

  bool operator==(TensorX const &other) const;
  bool operator!=(TensorX const &other) const;
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

OpX *create_opx(sl::Operator const &op,
                int parallel_degree,
                TensorX const &input1 = TensorX::NO_TX,
                TensorX const &input2 = TensorX::NO_TX,
                TensorX const &input3 = TensorX::NO_TX,
                TensorX const &input4 = TensorX::NO_TX);
void create_xfer(GraphXfer &xfer, sl::Rule const &r, int parallel_degree);
std::vector<GraphXfer*> create_xfers(
  FFModel *model, sl::RuleCollection const &rules, int parallel_degree);

class GraphCompare {
public:
  bool operator() (Graph* lhs, Graph* rhs) {
    return lhs->optimal_cost() > rhs->optimal_cost();
  }
};

class GraphXferMatch {
public:
  GraphXferMatch(GraphXfer const *);

  void add_mapping(Node const &, OpX*);
  void add_mapping(OpX*, Node const &);
  void add_input_mapping(int, std::pair<Node, int> const &);
  void add_output_mapping(TensorX const &, TensorX const &);
  OpX *at(Node const &) const;
  Node at(OpX *) const;
  void set_graph(Graph const *);

  bool containsNode(Graph const *, Node const &) const;
  bool containsEdge(Graph const *, Edge const &) const;

  GraphXfer const *get_xfer() const;
  std::unordered_set<Node> get_nodes() const;
private:
  std::map<Node, OpX*, NodeCompare> nodeToOpX;
  std::map<OpX*, Node> opXToNode;
  std::map<TensorX, TensorX, TensorXCompare> mappedOutputs;
  size_t graph_hash;
  GraphXfer const *xfer;
};

class GraphXfer {
public:
  GraphXfer(FFModel* _model);
  TensorX new_tensor(void);
  bool can_match(OpX* srcOp, const Node& op, Graph const *graph);
  void match(OpX* srcOp, const Node& op, Graph const *graph);
  void unmatch(OpX* srcOp, const Node& op, Graph const *graph);
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
  OpX* create_relu(const TensorX& input);
  OpX* create_linear(const TensorX& input,
                     const OpX* match_opx,
                     int num_dims,
                     ActiMode acti_mode,
                     bool use_bias);
  OpX* create_conv2d(const TensorX& input,
                     const OpX* match_opx);
  OpX* create_pool2d(const TensorX& input,
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

  
  Graph* create_new_graph(Graph const *graph, SimplificationSettings const &settings);
  bool create_new_operator(const OpX* opx, Node& op);

  std::string get_name() const;

  void run(int depth, Graph* graph,
           std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>&,
           std::unordered_set<size_t>&, float threshold, int maxNumOps, 
           SimplificationSettings const &simplification_settings,
           int& num_matches_found, int& num_matches_rejected);

  void find_matches(Graph const *, std::vector<GraphXferMatch>& matches);
  GraphXferMatch get_match_record(Graph const *) const;
private:
  void find_matches(int depth, Graph const *graph, std::vector<GraphXferMatch>& matches);
public:
  FFModel* model;
  tl::optional<std::string> name = tl::nullopt;
  int tensorId;
  std::map<Node, OpX*, NodeCompare> mappedOps;
  std::multimap<int, std::pair<Node, int>> mappedInputs;
  std::map<TensorX, TensorX, TensorXCompare> mappedOutputs;
  std::vector<OpX*> srcOps;
  std::vector<OpX*> dstOps;
};

class GraphSearchHelper {
public:
  GraphSearchHelper(FFModel *model);
  void graph_optimize(size_t budget, 
                      bool only_data_parallel,
                      std::unique_ptr<Graph>& best_graph,
                      std::unordered_map<Node, MachineView>& optimal_views);
  // Experimental. To be merged with graph_optimize eventually.
  void graph_optimize_with_memory(size_t budget, 
                      bool only_data_parallel,
                      std::unique_ptr<Graph>& best_graph,
                      std::unordered_map<Node, MachineView>& optimal_views);
  void graph_optimize_no_split(
      size_t budget, 
      bool only_data_parallel,
      std::unique_ptr<Graph>& best_graph,
      std::unordered_map<Node, MachineView>& optimal_views);
private:
  template <typename T>
  T generic_sequence_optimize(Graph const *graph,
                              Node const &sink_node,
                              tl::optional<ParallelTensorShape> const &output_shape,
                              tl::optional<ParallelTensorShape> const &input_shape);

  // Experimental. To be merged with generic_sequence_optimize eventually.
  template <typename T>
  T generic_sequence_optimize_with_memory(
      Graph const* graph, Node const& sink_node,
      tl::optional<ParallelTensorShape> const& output_shape,
      tl::optional<ParallelTensorShape> const& input_shape);

  float sequence_optimize(Graph const *graph, 
                          Node const &sink_node, 
                          tl::optional<ParallelTensorShape> const &output_shape, 
                          tl::optional<ParallelTensorShape> const &input_shape);

  template <typename T>
  T execute_sequence_split(std::unique_ptr<Graph> const &pre_graph,
                           std::unique_ptr<Graph> const &post_graph,
                           tl::optional<ParallelTensorShape> const &output_shape,
                           tl::optional<ParallelTensorShape> const &input_shape,
                           Node const &sink_node,
                           Node const &bottleneck, 
                           ParallelTensorShape const &bottleneck_output_shape);
  // Experimental. To be merged with execute_sequence_split().
  template <typename T>
  T execute_sequence_split_with_memory(std::unique_ptr<Graph> const &pre_graph,
                           std::unique_ptr<Graph> const &post_graph,
                           tl::optional<ParallelTensorShape> const &output_shape,
                           tl::optional<ParallelTensorShape> const &input_shape,
                           Node const &sink_node,
                           Node const &bottleneck, 
                           ParallelTensorShape const &bottleneck_output_shape);

  void generate_all_pcg_xfers();
  void load_graph_substitutions(std::vector<GraphXfer*> &xfers) const;
  Graph *construct_graph();
  void subgraph_optimize(Graph *subgraph);

  std::unique_ptr<Graph> base_optimize(Graph const *, SimplificationSettings const &simplification_settings);

  // Experimental. To be merged with base_optimize().
  std::unique_ptr<Graph> base_optimize_with_memory(Graph const *, SimplificationSettings const &simplification_settings);

  std::vector<ParallelTensorShape> possible_split_output_tensor_shapes(Node const &) const;
  
  void find_rewrite_matches(Graph const *graph, std::vector<GraphXferMatch>& matches) const;
  tl::optional<Node> find_split_node(Graph const *graph, int base_optimize_threshold) const;

  template <typename T>
  tl::optional<T> try_get_cost_from_cache(size_t hash) const;

  template <typename T>
  void try_cache_result(size_t hash, T const &value);

  template <typename T>
  T get_optimal_cost(std::unique_ptr<Graph> optimized) const;
private:
  std::unordered_map<size_t, float> cached_optimized_graphs;
  std::vector<GraphXfer*> all_pcg_xfers;
  FFModel* model;
  FFConfig const &config;
  MemoryOptimConfig mem_config;
  std::unique_ptr<RecursiveLogger> logger;
};

}; // namespace FlexFlow
#endif
