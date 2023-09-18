#include "doctest/doctest.h"
#include "op_task_invocation.h"
#include "op_task_signature.h"
#include "aggregate.h"
#include "op-attrs/ops/aggregate.h"
#include <variant>

using namespace FlexFlow;

void check_signature_binding(OpTaskSignature & signature, OpTaskBinding & binding) {
  // check tensors
  auto const& tensor_slots = signature.get_tensor_slots();
  auto const& tensor_bindings = binding.get_tensor_bindings();
  CHECK (tensor_slots.size() == tensor_bindings.size());
  for (OpTensorSlotSpec const& tensor_slot_spec: tensor_slots) {
    slot_id name = tensor_slot_spec.name;
    CHECK (tensor_bindings.count({name, tensor_slot_spec.is_grad}));
    OpTensorSpec const tensor_spec = tensor_bindings.at({name, tensor_slot_spec.is_grad});
    CHECK (tensor_spec.role == tensor_slot_spec.tensor_role);
  }

  // check arg types
  auto const& arg_types = signature.get_arg_types();
  auto const& arg_bindings = binding.get_arg_bindings();
  CHECK (arg_types.size() == arg_bindings.size());
  for (auto const &[slot_id, op_arg_spec] : arg_bindings) {
    std::type_index op_arg_spec_type = std::visit(OpArgSpecTypeAccessor(), op_arg_spec);
    CHECK (arg_types.count(slot_id));
    CHECK (arg_types.at(slot_id) == op_arg_spec_type);
  }
}

TEST_CASE("OpTaskInvocation:Aggregate") {
  // get binding from operator forward
  OpTaskSignature signature = fwd_signature<AGGREGATE_FWD_TASK_ID>();
  AggregateAttrs attrs {n: 12, lambda_bal: 1.0};
  OpTaskBinding binding = forward(attrs).binding;
  check_signature_binding(signature, binding);
}

TEST_CASE("OpTaskBinding:Aggregate") {
  OpTaskBinding fwd;

  fwd.bind(0, input_tensor(0));
  fwd.bind(1, input_tensor(1));
  fwd.bind(2, input_tensor(2));

  bool profiling = true;
  AggregateAttrs attrs {n: 12, lambda_bal: 1.0};
  fwd.bind_arg<bool>(3, profiling);
  fwd.bind_arg<AggregateAttrs>(4, attrs);

  OpTaskBinding bwd = infer_bwd_binding(fwd);

  OpTaskBinding correct_bwd;

  correct_bwd.bind(0, input_tensor(0));
  correct_bwd.bind_grad(0, input_tensor(0));
  correct_bwd.bind(1, input_tensor(1));
  correct_bwd.bind_grad(1, input_tensor(1));
  correct_bwd.bind(2, input_tensor(2));
  correct_bwd.bind_grad(2, input_tensor(2));

  correct_bwd.bind_arg<bool>(3, profiling);
  correct_bwd.bind_arg<AggregateAttrs>(4, attrs);

  CHECK(bwd == correct_bwd);
}