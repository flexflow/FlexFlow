#ifndef _FLEXFLOW_NODE_CACHE_H
#define _FLEXFLOW_NODE_CACHE_H

namespace FlexFlow {

class NodeCache {
public:
  typedef PCG::Node Node;
  Node get_or_create_noop_node(const Tensor input);
  Node get_or_create_input_node(const TensorShape&);
  Node get_or_create_concat_node(int num_inputs,
                                 const Tensor* inputs,
                                 int axis);
  Node get_or_create_element_binary_node(const Tensor input1,
                                         const Tensor input2,
                                         OperatorType type);
  Node get_or_create_embedding_node(const Tensor input,
                                    int num_entries,
                                    int out_channels,
                                    AggrMode aggr);
  Node get_or_create_linear_node(const Tensor input,
                                 int out_dim,
                                 ActiMode activation,
                                 bool use_bias);
  Node get_or_create_linear_node(const Tensor input,
                                 const LinearParams& params);
  Node get_or_create_multihead_attn_node(const Tensor query,
                                         const Tensor key,
                                         const Tensor value,
                                         int embed_dim,
                                         int num_heads,
                                         int kdim,
                                         int vdim,
                                         float dropout,
                                         bool bias,
                                         bool add_bias_kv,
                                         bool add_zero_attn);
  Node get_or_create_softmax_node(const Tensor input,
                                  int softmax_dim);
  Node get_or_create_repartition_node(const Tensor input,
                                      int repartition_dim,
                                      int repartition_degree);
  Node get_or_create_replicate_node(const Tensor input,
                                    int replicate_dim,
                                    int replicate_degree);
  Node get_or_create_reduction_node(const Tensor input,
                                    int reduction_dim,
                                    int reduction_degree);
  Node get_or_create_combine_node(const Tensor input,
                                  int combine_dim,
                                  int combine_degree);
  Node get_or_create_fused_parallel_node(const Tensor input,
                                         const std::vector<ParallelOpInfo>& parallel_ops);
  Node get_or_create_conv2d_node(const Tensor input, 
                                 int out_channels,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w, 
                                 int padding_h, int padding_w,
                                 ActiMode activation, 
                                 int groups,
                                 bool use_bias);
  Node get_or_create_conv2d_node(const Tensor input,
                                 const Conv2DParams& params);
  Node get_or_create_pool2d_node(const Tensor input,
                                 int kernelH, int kernelW,
                                 int strideH, int strideW,
                                 int paddingH, int paddingW,
                                 PoolType type,
                                 ActiMode activation);
  Node get_or_create_pool2d_node(const Tensor input,
                                 const Pool2DParams& params);
  Node get_or_create_flat_node(const Tensor input);
  Node get_or_create_element_unary_node(const Tensor input,
                                        OperatorType type,
                                        bool inplace, 
                                        float scalar);
  Node get_or_create_parallel_op_node(const Tensor input, 
                                      ParallelOpInfo const &);
private:
  std::unordered_map<size_t, NoOp*> cached_noop_ops;
  std::unordered_map<size_t, NoOp*> cached_input_ops;
  std::unordered_map<size_t, Concat*> cached_concat_ops;
  std::unordered_map<size_t, ElementBinary*> cached_element_binary_ops;
  std::unordered_map<size_t, ElementUnary*> cached_element_unary_ops;
  std::unordered_map<size_t, Embedding*> cached_embedding_ops;
  std::unordered_map<size_t, Linear*> cached_linear_ops;
  std::unordered_map<size_t, Conv2D*> cached_conv2d_ops;
  std::unordered_map<size_t, Pool2D*> cached_pool2d_ops;
  std::unordered_map<size_t, Flat*> cached_flat_ops;
  std::unordered_map<size_t, MultiHeadAttention*> cached_multihead_attn_ops;
  std::unordered_map<size_t, Softmax*> cached_softmax_ops;
  std::unordered_map<size_t, Repartition*> cached_repartition_ops;
  std::unordered_map<size_t, Replicate*> cached_replicate_ops;
  std::unordered_map<size_t, Reduction*> cached_reduction_ops;
  std::unordered_map<size_t, Combine*> cached_combine_ops;
  std::unordered_map<size_t, FusedParallelOp*> cached_fused_parallel_ops;

  int op_global_guid = 3000;
};

}; // namespace

#endif // _FLEXFLOW_NODE_CACHE_H
