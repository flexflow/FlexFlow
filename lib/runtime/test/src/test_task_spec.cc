#include "doctest/doctest.h"
#include "task_spec.h"

using namespace FlexFlow;

TEST_CASE("OpTaskSignature") {
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
