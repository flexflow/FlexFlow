#include "doctest/doctest.h"
#include "task_spec.h"

using namespace FlexFlow;

TEST_CASE("OpTaskSignature,Binding") {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(0);
  fwd.add_input_slot(1);
  fwd.add_output_slot(2);

  OpTaskSignature bwd = infer_bwd_signature(fwd);

  OpTaskSignature correct_bwd = {OpTaskType::BWD};

  correct_bwd.add_input_slot(0);
  correct_bwd.add_input_grad_slot(0);
  correct_bwd.add_input_slot(1);
  correct_bwd.add_input_grad_slot(1);
  correct_bwd.add_output_slot(2);
  correct_bwd.add_output_grad_slot(2);

  CHECK(bwd == correct_bwd);


}

TEST_CASE("OpTaskBinding") {
  OpTaskBinding fwd;

  binding.bind(0, input_tensor(0));
  binding.bind(1, input_tensor(1));
  binding.bind(2, input_tensor(2));

  OpTaskBinding bwd = infer_bwd_binding(fwd);

  OpTaskBinding correct_bwd;

  correct_bwd.bind(0);
  correct_bwd.bind_grad(0);
}