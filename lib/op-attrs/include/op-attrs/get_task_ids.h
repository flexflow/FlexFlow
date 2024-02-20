#ifndef _FLEXFLOW_OP_ATTRS_GET_OP_TYPE_H
#define _FLEXFLOW_OP_ATTRS_GET_OP_TYPE_H

#include "operator_attrs.h"
#include "tasks.h"
#include "utils/variant.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(BatchMatmulAttrs const &);
std::vector<task_id_t> get_task_ids(BatchNormAttrs const &);
std::vector<task_id_t> get_task_ids(BroadcastAttrs const &);
std::vector<task_id_t> get_task_ids(CastAttrs const &);
std::vector<task_id_t> get_task_ids(ConcatAttrs const &);
std::vector<task_id_t> get_task_ids(Conv2DAttrs const &);
std::vector<task_id_t> get_task_ids(DropoutAttrs const &);
std::vector<task_id_t> get_task_ids(ElementBinaryAttrs const &);
std::vector<task_id_t> get_task_ids(ElementScalarUnaryAttrs const &);
std::vector<task_id_t> get_task_ids(ElementUnaryAttrs const &);
std::vector<task_id_t> get_task_ids(EmbeddingAttrs const &);
std::vector<task_id_t> get_task_ids(FlatAttrs const &);
std::vector<task_id_t> get_task_ids(GatherAttrs const &);
std::vector<task_id_t> get_task_ids(InputAttrs const &);
std::vector<task_id_t> get_task_ids(LayerNormAttrs const &);
std::vector<task_id_t> get_task_ids(LinearAttrs const &);
std::vector<task_id_t> get_task_ids(MultiHeadAttentionAttrs const &);
std::vector<task_id_t> get_task_ids(NoopAttrs const &);
std::vector<task_id_t> get_task_ids(Pool2DAttrs const &);
std::vector<task_id_t> get_task_ids(ReduceAttrs const &);
std::vector<task_id_t> get_task_ids(ReshapeAttrs const &);
std::vector<task_id_t> get_task_ids(ReverseAttrs const &);
std::vector<task_id_t> get_task_ids(SplitAttrs const &);
std::vector<task_id_t> get_task_ids(SoftmaxAttrs const &);
std::vector<task_id_t> get_task_ids(TopKAttrs const &);
std::vector<task_id_t> get_task_ids(TransposeAttrs const &);
std::vector<task_id_t> get_task_ids(CombineAttrs const &);
std::vector<task_id_t> get_task_ids(ReductionAttrs const &);
std::vector<task_id_t> get_task_ids(RepartitionAttrs const &);
std::vector<task_id_t> get_task_ids(ReplicateAttrs const &);

struct GetTaskIdsFunctor {
  template <typename T>
  std::vector<task_id_t> operator()(T const &t) {
    return get_task_ids(t);
  }
};

template <typename... Ts>
std::vector<task_id_t> get_task_ids(variant<Ts...> const &attrs) {
  return visit(GetTaskIdsFunctor{}, attrs);
}

} // namespace FlexFlow

#endif
