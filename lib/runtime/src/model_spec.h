#ifndef _FLEXFLOW_RUNTIME_SRC_MODEL_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_MODEL_SPEC_H

#include "tensor.h"
#include "layer.h"

namespace FlexFlow {

struct ModelSpec {
public:
  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(Tensor const &, std::string const &name);
  // Add an add layer
  Tensor add(Tensor const &x,
             Tensor const &y,
             std::string const &name);
  // Add a subtract layer
  Tensor subtract(Tensor const &x,
                  Tensor const &y,
                  std::string const &name);
  // Add a multiply layer
  Tensor multiply(Tensor const &x,
                  Tensor const &y,
                  std::string const &name);
  // Add a divide layer
  Tensor divide(Tensor const &x,
                Tensor const &y,
                std::string const &name);
  // Add a max layer
  Tensor max(Tensor const &x,
             Tensor const &y,
             std::string const &name);
  // Add a min layer
  Tensor min(Tensor const &x,
             Tensor const &y,
             std::string const &name);
  // Add a rsqrt layer
  Tensor rsqrt(Tensor const &x, std::string const &name);
  // Add a pow layer
  Tensor pow(Tensor const &x,
             float exponent,
             std::string const &name);
  // Add a scalar multiply layer
  Tensor scalar_multiply(Tensor const &x,
                         float scalar,
                         std::string const &name);
  Tensor scalar_add(Tensor const &x,
                    float scalar,
                    std::string const &name);
  Tensor scalar_sub(Tensor const &lhs,
                    float rhs,
                    std::string const &name);
  Tensor scalar_truediv(Tensor const &numerator,
                        float denominator,
                        std::string const &name);
  // Add a sin layer
  Tensor sin(Tensor const &x, std::string const &name);
  // Add a cos layer
  Tensor cos(Tensor const &x, std::string const &name);
  // Add an activation layer
  Tensor relu(Tensor const &x, std::string const &name);
  Tensor identity(Tensor const &x, std::string const &name);
  Tensor gelu(Tensor const &x, std::string const &name);
  Tensor sigmoid(Tensor const &x, std::string const &name);
  Tensor tanh(Tensor const &x, std::string const &name);
  Tensor elu(Tensor const &x, std::string const &name);
  // Add a 2D convolutional layer
  Tensor conv2d(Tensor const &input,
                int outChannels,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                std::string const &name,
                ActiMode activation = AC_MODE_NONE,
                int groups = 1,
                bool use_bias = true,
                Initializer *kernel_initializer = nullptr,
                Initializer *bias_initializer = nullptr);
  // Add a dropout layer
  Tensor dropout(Tensor const &input,
                 float rate,
                 unsigned long long seed = 0,
                 char const *name = nullptr);
  // Add an embedding layer
  Tensor embedding(Tensor const &input,
                   int num_entires,
                   int outDim,
                   AggrMode aggr,
                   DataType dtype = DT_FLOAT,
                   Layer const *shared_op = nullptr,
                   Initializer *kernel_initializer = nullptr,
                   char const *name = nullptr);
  // Add a gather layer
  Tensor gather(Tensor const &input,
                Tensor const &index,
                int dim,
                char const *name = nullptr);
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
  Tensor aggregate(Tensor const *inputs,
                   int n,
                   float lambda_bal,
                   char const *name = nullptr);
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
  Tensor cast(Tensor const &input, DataType dtype, std::string const &name);
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
  Tensor create_tensor(LegionTensorShape const &shape,
                       bool create_grad = true);
  Tensor create_tensor(TensorShape const &,
                       bool create_grad = true);
private:
  void add_layer(Layer const &layer, std::vector<Tensor> const &inputs, std::vector<Tensor> const &weights, std::vector<Tensor> const &outputs);

  Tensor as_type(Tensor const &, DataType, std::string const &);

  std::pair<bool, Tensor> broadcast(Tensor const &, TensorShape const &target_shape);

  TensorShape get_broadcast_target_shape(std::vector<TensorShape> const &);

  Tensor element_binary(OperatorType, Tensor const &lhs, Tensor const &rhs, std::string const &name);

  Tensor element_unary(OperatorType, Tensor const &input, std::string const &name);
  Tensor element_scalar_unary(OperatorType, Tensor const &input, float scalar, std::string const &name);
  Tensor element_unary(variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &, Tensor const &input, std::string const &name);
private:
  LayerManager layer_mgr; 
  TensorManager tensor_mgr;
  LabelledOpenMultiDiGraph<Layer, Tensor> graph;
};

}

#endif
