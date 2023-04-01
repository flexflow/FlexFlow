#include "op-attrs/operator_attrs.h"
#include "op-attrs/ffconst_utils.h"
#include "utils/record_formatter.h"
#include "visit_struct/visit_struct.hpp"
#include "op-attrs/ffconst_utils.h"
#include "utils/containers.h"

namespace FlexFlow {

/* OperatorType GetOpType::operator()(BatchMatmulAttrs const &p) const { return OP_BATCHMATMUL; } */
/* OperatorType GetOpType::operator()(Conv2DAttrs const &p) const { return OP_CONV2D; } */
/* OperatorType GetOpType::operator()(ConcatAttrs const &p) const { return OP_CONCAT; } */
/* OperatorType GetOpType::operator()(CastAttrs const &p) const { return OP_CAST; } */
/* OperatorType GetOpType::operator()(ElementBinaryAttrs const &p) const { return p.type; } */
/* OperatorType GetOpType::operator()(ElementUnaryAttrs const &p) const { return p.op_type; } */
/* OperatorType GetOpType::operator()(DropoutAttrs const &p) const { return OP_DROPOUT; } */
/* OperatorType GetOpType::operator()(EmbeddingAttrs const &p) const { return OP_EMBEDDING; } */
/* OperatorType GetOpType::operator()(FlatAttrs const &p) const { return OP_FLAT; } */
/* OperatorType GetOpType::operator()(LayerNormAttrs const &p) const { return OP_LAYERNORM; } */
/* OperatorType GetOpType::operator()(LinearAttrs const &p) const { return OP_LINEAR; } */
/* OperatorType GetOpType::operator()(MultiHeadAttentionAttrs const &p) const { return OP_DROPOUT; } */
/* OperatorType GetOpType::operator()(Pool2DAttrs const &p) const { return OP_POOL2D; } */
/* OperatorType GetOpType::operator()(ReshapeAttrs const &p) const { return OP_RESHAPE; } */
/* OperatorType GetOpType::operator()(SplitAttrs const &p) const { return OP_SPLIT; } */
/* OperatorType GetOpType::operator()(SoftmaxAttrs const &p) const { return OP_SOFTMAX; } */
/* OperatorType GetOpType::operator()(TransposeAttrs const &p) const { return OP_TRANSPOSE; } */
/* OperatorType GetOpType::operator()(RepartitionAttrs const &p) const { return OP_REPARTITION; } */
/* OperatorType GetOpType::operator()(ReplicateAttrs const &p) const { return OP_REPLICATE; } */
/* OperatorType GetOpType::operator()(ReductionAttrs const &p) const { return OP_REDUCTION; } */
/* OperatorType GetOpType::operator()(CombineAttrs const &p) const { return OP_COMBINE; } */
/* OperatorType GetOpType::operator()(FusedParallelOpAttrs const &p) const { return OP_FUSED_PARALLEL; } */

/* struct AsOpAttrs { */
/*   template <typename T> */
/*   OpAttrsInterface const &operator()(T const &p) { */
/*     return p; */
/*   } */
/* }; */

/* OperatorType get_op_type(OpAttrsInterface const &o) { */
/*   return o.op_type(); */
/* } */
/*                                                           // */
/* OperatorType get_op_type(CompGraphOperatorAttrs const &o) { */ 
/*   return get_op_type(visit(AsOpAttrs{}, o)); */
/* } */

/* OperatorType get_op_type(PCGOperatorAttrs const &o) { */ 
/*   return get_op_type(visit(AsOpAttrs{}, o)); */
/* } */

/* std::vector<ParallelTensorShape> get_output_shapes(PCGOperatorAttrs const &op_params, std::vector<ParallelTensorShape> const &input_tensor_shapes) { */
/*   return mpark::visit(AsOpAttrs{}, op_params).output_shapes(input_tensor_shapes); */
/* } */

/* bool is_parallel_op(PCGOperatorAttrs const &o) { */
/*   return is_parallel_op(get_op_type(o)); */
/* } */

void as_dot(int x, RecordFormatter &r) {
  r << std::to_string(x); 
}

void as_dot(std::string const &s, RecordFormatter &r) {
  r << s;
}

template <typename T>
void as_dot(std::vector<T> const &x, RecordFormatter &r) {
  RecordFormatter rr;
  for (T const &t : x) {
    as_dot(t, r);
  }
  r << rr;
}

void as_dot(ParallelOpInfo const &p, RecordFormatter &r) {
  RecordFormatter rr;
  as_dot(p.op_type, rr);
  as_dot(p.parallel_dim, rr);
  as_dot(p.parallel_degree, rr);
  r << rr; 
}

struct as_dot_visitor {
  RecordFormatter result;

  template <typename T>
  void operator()(const char *name, T const &t) {
    RecordFormatter kv;
    kv << name;
    as_dot(t, result);
    result << kv;
  }

  template <typename T>
  void operator()(T const &t) {
    as_dot(t, result);
  }

  /* template <typename V> */
  /* void operator()(const char *name, std::vector<V> const &t) { */
  /*   RecordFormatter kv; */
  /*   kv << name; */
  /*   RecordFormatter v; */
  /*   for (V const &vv : t) { */
  /*     v << as_dot_str(vv); */
  /*   } */
  /*   kv << v; */
  /* } */
};

template <typename T>
RecordFormatter generic_as_dot(T const &t) {
  as_dot_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

template <>
RecordFormatter generic_as_dot<FlatAttrs>(FlatAttrs const &p) {
  RecordFormatter r; 
  return r;
}

struct AsDot {
  template <typename T>
  RecordFormatter operator()(T const &t) {
    return generic_as_dot(t);
  }
};

RecordFormatter as_dot(PCGOperatorAttrs const &o) {
  return mpark::visit(AsDot{}, o);
}

/* int num_outputs(OperatorParameters const &o) { */
/*   switch (get_op_type(o)) { */
/*     case OP_SPLIT: */
/*   } */
/* } */

//tl::optional<OperatorParameters> get_op_parameters(Op const *op) {
//  switch (op->op_type) {
//    case OP_LINEAR:
//      return ((Linear *)op)->get_params();
//    case OP_CONV2D:
//      return ((Conv2D *)op)->get_params();
//    case OP_EW_ADD:
//    case OP_EW_SUB:
//    case OP_EW_MUL:
//    case OP_EW_DIV:
//      return ((ElementBinary *)op)->get_params();
//    case OP_EXP:
//    case OP_SIN:
//    case OP_COS:
//    case OP_SCALAR_MULTIPLY:
//    case OP_SCALAR_ADD:
//    case OP_SCALAR_SUB:
//    case OP_SCALAR_TRUE_DIV:
//    case OP_RELU:
//    case OP_SIGMOID:
//    case OP_TANH:
//    case OP_IDENTITY:
//    case OP_GELU:
//    case OP_ELU:
//      return ((ElementUnary *)op)->get_params();
//    case OP_CONCAT:
//      return ((Concat *)op)->get_params();
//    case OP_POOL2D:
//      return ((Pool2D *)op)->get_params();
//    case OP_CAST:
//      return ((Cast *)op)->get_params();
//    case OP_DROPOUT:
//      return ((Dropout *)op)->get_params();
//    case OP_EMBEDDING:
//      return ((Embedding *)op)->get_params();
//    case OP_FLAT:
//      return ((Flat *)op)->get_params();
//    case OP_MULTIHEAD_ATTENTION:
//      return ((MultiHeadAttention *)op)->get_params();
//    case OP_LAYERNORM:
//      return ((LayerNorm *)op)->get_params();
//    case OP_RESHAPE:
//      return ((Reshape *)op)->get_params();
//    case OP_SOFTMAX:
//      return ((Softmax *)op)->get_params();
//    case OP_REPARTITION:
//      return ((Repartition *)op)->get_params();
//    case OP_REPLICATE:
//      return ((Replicate *)op)->get_params();
//    case OP_REDUCTION:
//      return ((Reduction *)op)->get_params();
//    case OP_COMBINE:
//      return ((Combine *)op)->get_params();
//    case OP_FUSED_PARALLEL:
//      return ((FusedParallelOp *)op)->get_params();
//    case OP_TRANSPOSE:
//      return ((Transpose *)op)->get_params();
//    case OP_BATCHMATMUL:
//      return ((BatchMatmul *)op)->get_params();
//    case OP_SPLIT:
//      return ((Split *)op)->get_params();
//
//      // TODO: implement the get_params() function for the operators below and
//      // uncomment the lines below
//
//      // case OP_NOOP:
//      //   return ((NoOp *)op)->get_params();
//      // case OP_TOPK:
//      //   return ((TopK *)op)->get_params();
//      // case OP_MEAN:
//      //   return ((Mean *)op)->get_params();
//      // case OP_GROUP_BY:
//      //   return ((Group_by *)op)->get_params();
//      // case OP_CACHE:
//      //   return ((Cache *)op)->get_params();
//      // case OP_AGGREGATE:
//      //   return ((Aggregate *)op)->get_params();
//      // case OP_AGG_SPEC:
//      //   return ((AggregateSpec *)op)->get_params();
//      // case OP_REVERSE:
//      //   return ((Reverse *)op)->get_params();
//      // case OP_BATCHNORM:
//      //   return ((BatchNorm *)op)->get_params();
//
//    default:
//      return tl::nullopt;
//  }
//}

}

