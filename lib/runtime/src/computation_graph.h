#ifndef _FLEXFLOW_RUNTIME_SRC_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_COMPUTATION_GRAPH_H

#include "tensor.h"
#include "layer.h"
#include "utils/expected.h"
#include "utils/graph.h"

namespace FlexFlow {

template <typename T>
using or_error_msg = expected<T, std::string>;

struct TensorSourceInfo {
  Layer layer;
  int idx;
};

struct tensor_guid_t : strong_typedef<tensor_guid_t, Node> {
  using strong_typedef::strong_typedef;  
};

struct ComputationGraph {
public:
  ComputationGraph() = default;
  ComputationGraph(ComputationGraph const &) = default;
  ComputationGraph(ComputationGraph &&) = default;
  
  ComputationGraph &operator=(ComputationGraph const &) = default;

  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(Tensor const &, optional<std::string> const &name = nullopt);
  // Add an add layer
  Tensor add(Tensor const &x,
             Tensor const &y,
             optional<std::string> const &name = nullopt);
  // Add a subtract layer
  Tensor subtract(Tensor const &x,
                  Tensor const &y,
                  optional<std::string> const &name = nullopt);
  // Add a multiply layer
  Tensor multiply(Tensor const &x,
                  Tensor const &y,
                  optional<std::string> const &name = nullopt);
  // Add a divide layer
  Tensor divide(Tensor const &x,
                Tensor const &y,
                optional<std::string> const &name = nullopt);
  // Add a max layer
  Tensor max(Tensor const &x,
             Tensor const &y,
             optional<std::string> const &name = nullopt);
  // Add a min layer
  Tensor min(Tensor const &x,
             Tensor const &y,
             optional<std::string> const &name = nullopt);
  // Add a rsqrt layer
  Tensor rsqrt(Tensor const &x, 
               optional<std::string> const &name = nullopt);
  // Add a pow layer
  Tensor pow(Tensor const &x,
             float exponent,
             optional<std::string> const &name = nullopt);
  // Add a scalar multiply layer
  Tensor scalar_multiply(Tensor const &x,
                         float scalar,
                         optional<std::string> const &name = nullopt);
  Tensor scalar_add(Tensor const &x,
                    float scalar,
                    optional<std::string> const &name = nullopt);
  Tensor scalar_sub(Tensor const &lhs,
                    float rhs,
                    optional<std::string> const &name = nullopt);
  Tensor scalar_truediv(Tensor const &numerator,
                        float denominator,
                        optional<std::string> const &name = nullopt);
  // Add a sin layer
  Tensor sin(Tensor const &x, 
             optional<std::string> const &name = nullopt);
  // Add a cos layer
  Tensor cos(Tensor const &x, optional<std::string> const &name = nullopt);
  // Add an activation layer
  Tensor relu(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor identity(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor gelu(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor sigmoid(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor tanh(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor elu(Tensor const &x, optional<std::string> const &name = nullopt);
  // Add a 2D convolutional layer
  Tensor conv2d(Tensor const &input,
                int outChannels,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                ActiMode activation = AC_MODE_NONE,
                int groups = 1,
                bool use_bias = true,
                Initializer *kernel_initializer = nullptr,
                Initializer *bias_initializer = nullptr,
                optional<std::string> const &name = nullopt);
  // Add a dropout layer
  Tensor dropout(Tensor const &input,
                 float rate,
                 unsigned long long seed = 0,
                 optional<std::string> const &name = nullopt);
  // Add an embedding layer
  Tensor embedding(Tensor const &input,
                   int num_entries,
                   int outDim,
                   AggrMode aggr,
                   DataType dtype = DT_FLOAT,
                   Initializer *kernel_initializer = nullptr,
                   optional<std::string> const &name = nullopt);
  // Add a gather layer
  std::vector<Tensor> gather(Tensor const &input,
                Tensor const &index,
                ff_dim_t dim,
                optional<std::string> const &name = nullopt);
  // Add a group_by layer
  void group_by(Tensor const &data,
                Tensor const &assign,
                Tensor *outputs,
                int n,
                float alpha,
                char const *name = nullptr);
  // Add a cache layer
  Tensor cache(Tensor const &input,
               int num_batches,
               std::function<float(float *, void const *, void const *, int)>
                   score_f = {},
               char const *name = nullptr);
  // Add aggregate layer
  Tensor aggregate(Tensor const &gate_preds,
                   Tensor const &gate_assign,
                   Tensor const &true_gate_assign,
                   Tensor const &full_gate_gradients,
                   std::vector<Tensor> const &exp_preds,
                   int n,
                   float lambda_bal,
                   optional<std::string> const &maybe_name);
  // Add aggregate_spec layer
  Tensor aggregate_spec(Tensor const *inputs,
                        int n,
                        float lambda_bal,
                        char const *name = nullptr);
  // Add a 2D pooling layer
  Tensor pool2d(Tensor const &input,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE,
                char const *name = nullptr);
  // Add a batch_norm layer
  Tensor layer_norm(Tensor const &input,
                    std::vector<int> const &axes,
                    bool elementwise_affine,
                    float eps,
                    char const *name = nullptr);
  // Add a batch_norm layer
  Tensor
      batch_norm(Tensor const &input, bool relu = true, char const *name = nullptr);
  // Add a batch_matmul layer
  Tensor batch_matmul(Tensor const &A,
                      Tensor const &B,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      char const *name = nullptr);
  // Add a dense layer
  Tensor dense(Tensor const &input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               DataType data_type = DT_FLOAT,
               Layer const *shared_op = nullptr,
               Initializer *kernel_initializer = nullptr,
               Initializer *bias_initializer = nullptr,
               char const *name = nullptr);
  // Add a cast layer
  Tensor cast(Tensor const &input, DataType dtype, optional<std::string> const &name = nullopt);
  // Add a concat layer
  Tensor
      concat(int n, Tensor const *tensors, int axis, char const *name = nullptr);
  // Add a mean layer
  Tensor mean(Tensor const &input,
              std::vector<int> const &dims,
              bool keepdims,
              char const *name);
  // Add a moe layer (wrapping topk, group_by and aggregate operators)
  Tensor moe(Tensor const &input,
             int num_exp,
             int num_select,
             int expert_hidden_size,
             float alpha,
             float lambda);
  // Add a split layer
  void split(Tensor const &input,
             Tensor *outputs,
             std::vector<int> const &split,
             int axis,
             char const *name = nullptr);
  // Add a flat layer
  Tensor flat(Tensor const &input, char const *name = nullptr);
  // Add a softmax layer
  Tensor softmax(Tensor const &input, int dim = -1, char const *name = nullptr);
  // Create input tensors and constants
  Tensor transpose(Tensor const &input,
                   std::vector<int> const &perm,
                   char const *name = nullptr);
  Tensor reduce_sum(Tensor const &input,
                    std::vector<int> const &axes,
                    bool keepdims = false,
                    char const *name = nullptr);
  Tensor reshape(Tensor const &input,
                 std::vector<int> const &shape,
                 char const *name = nullptr);
  Tensor reverse(Tensor const &input, int axis, char const *name = nullptr);
  void top_k(Tensor const &input,
             Tensor *outputs,
             int k,
             bool sorted,
             char const *name = nullptr);
  Tensor multihead_attention(Tensor const &query,
                             Tensor const &key,
                             Tensor const &value,
                             int embed_dim,
                             int num_heads,
                             int kdim = 0,
                             int vdim = 0,
                             float dropout = 0.0f,
                             bool bias = true,
                             bool add_bias_kv = false,
                             bool add_zero_attn = false,
                             Initializer *kernel_initializer = nullptr,
                             char const *name = nullptr);
  Tensor create_tensor(TensorShape const &,
                       bool create_grad = true);
  Tensor create_tensor(LegionTensorShape const &shape,
                       bool create_grad = true);
  Parameter create_weight(TensorShape const &, 
                          bool create_grad = true,
                          Initializer *initializer = nullptr,
                          ParameterSyncType sync_type = ParameterSyncType::NONE);
  Parameter create_weight(LegionTensorShape const &,
                          bool create_grad = true,
                          Initializer *initializer = nullptr,
                          ParameterSyncType sync_type = ParameterSyncType::NONE);

  optional<TensorSourceInfo> get_source(Tensor const &) const;

  std::vector<Tensor> get_outputs(Layer const &) const;
  Tensor get_output(Layer const &, int idx) const;

  Tensor at(MultiDiEdge const &) const; 
  Layer at(Node const &) const;

  friend void swap(ComputationGraph &, ComputationGraph &);
private:
  Tensor broadcast(Tensor const &, TensorShape const &);

  void add_layer(Layer const &layer, std::vector<Tensor> const &inputs, std::vector<Tensor> const &weights, std::vector<Tensor> const &outputs);
  Tensor add_layer(Layer const &layer, 
                 std::vector<Tensor> const &inputs, 
                 std::vector<std::pair<TensorShape, Initializer *>> const &weight_shapes,
                 TensorShape const &output_shape);
  std::vector<Tensor> add_layer(Layer const &layer, 
                 std::vector<Tensor> const &inputs, 
                 std::vector<std::pair<TensorShape, Initializer *>> const &weight_shapes,
                 std::vector<TensorShape> const &output_shapes);

  Tensor as_type(Tensor const &, DataType, std::string const &);

  TensorShape get_broadcast_target_shape(std::vector<TensorShape> const &);

  Tensor element_binary(OperatorType, Tensor const &lhs, Tensor const &rhs, optional<std::string> const &name = nullopt);

  Tensor element_unary(OperatorType, Tensor const &input, optional<std::string> const &name = nullopt);
  Tensor element_scalar_unary(OperatorType, Tensor const &input, float scalar, optional<std::string> const &name = nullopt);
  Tensor element_unary(variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &, Tensor const &input, optional<std::string> const &name = nullopt);
private:
  OutputLabelledMultiDiGraph<Layer, Tensor> graph;
};

static_assert(std::is_copy_constructible<ComputationGraph>::value, "");
static_assert(std::is_move_constructible<ComputationGraph>::value, "");
static_assert(std::is_copy_assignable<ComputationGraph>::value, "");
static_assert(std::is_copy_constructible<ComputationGraph>::value, "");

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::tensor_guid_t);

#endif
