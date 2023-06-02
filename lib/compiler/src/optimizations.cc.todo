#include "optimizations.h"
#include "op-attrs/get_op_type.h"
#include "utils/graph/rewriting.h"

namespace FlexFlow {

struct OptimizeUnnecessaryGradientCalculations {
  OptimizeUnnecessaryGradientCalculations() = default;

  Operator operator()(LabelledOpenMultiDiGraph<Operator, ParallelTensorAttrs> const &g, 
                      Node const &n, 
                      Operator const &op) { return op; }
  ParallelTensorAttrs operator()(LabelledOpenMultiDiGraph<Operator, ParallelTensorAttrs> const &g,
                                 MultiDiEdge const &e, 
                                 ParallelTensorAttrs const &pt) {
    ParallelTensorAttrs result = pt;
    if (get_op_type(g.at(e.src).attrs) == OperatorType::INPUT) {
      result.create_gradients = CreateGrad::NO;
    }
    return result;
  }
};

ParallelComputationGraph optimize_unnecessary_gradient_calculations(ParallelComputationGraph const &pcg) {
  // If an operator's input is training data
  // No need to compute its gradients
  return pcg.on_underlying([](LabelledOpenMultiDiGraph<Operator, ParallelTensor> const &g) {
    return rewrite(OptimizeUnnecessaryGradientCalculations{}, g);
  });
}

ParallelComputationGraph enable_inplace_operators(ParallelComputationGraph const &pcg) {
  
}

void FFModel::perform_fusion_optimizations() {
  fprintf(stderr, "Applying fusion optimizations during compilation...\n");
  fprintf(stderr, "%zu operators before fusion...\n", operators.size());
  std::vector<Op *> new_operators;
  std::vector<Op *> old_operators = operators;
  while (apply_fusion(operators, new_operators)) {
    for (size_t i = 0; i < new_operators.size(); i++) {
      for (int idx = 0; idx < new_operators[i]->numInputs; idx++) {
        for (size_t j = i + 1; j < new_operators.size(); j++) {
          if (new_operators[i]->inputs[idx]->owner_op == new_operators[j]) {
            assert(false);
          }
        }
      }
    }
    operators = new_operators;
  }
  // Check integrity
  for (size_t l = 0; l < operators.size(); l++) {
    if (operators[l]->op_type == OP_FUSED) {
      FusedOp *fused = (FusedOp *)operators[l];
      int ioff = 0, woff = 0, ooff = 0;
      for (int op = 0; op < fused->numOperators; op++) {
        Op *old_op = fused->operators[op];
        for (int i = 0; i < fused->op_num_inputs[op]; i++) {
          int my_off = fused->op_input_idx[i + ioff];
          if (fused->op_input_source[i + ioff] == FusedOp::SOURCE_INPUT) {
            assert(fused->inputs[my_off]->region ==
                   old_op->inputs[i]->region);
          } else if (fused->op_input_source[i + ioff] ==
                     FusedOp::SOURCE_OUTPUT) {
            assert(fused->outputs[my_off]->region ==
                   old_op->inputs[i]->region);
          } else {
            assert(false);
          }
        }
        for (int i = 0; i < fused->op_num_weights[op]; i++) {
          int my_off = fused->op_weight_idx[i + woff];
          assert(fused->op_weight_source[i + woff] == FusedOp::SOURCE_WEIGHT);
          assert(fused->weights[my_off]->region ==
                 old_op->weights[i]->region);
        }
        for (int i = 0; i < fused->op_num_outputs[op]; i++) {
          int my_off = fused->op_output_idx[i + ooff];
          assert(fused->op_output_source[i + ooff] == FusedOp::SOURCE_OUTPUT);
          assert(fused->outputs[my_off]->region ==
                 old_op->outputs[i]->region);
        }
        ioff += fused->op_num_inputs[op];
        woff += fused->op_num_weights[op];
        ooff += fused->op_num_outputs[op];
      }
    } else {
      bool found = false;
      for (size_t i = 0; i < old_operators.size(); i++) {
        if (old_operators[i] == operators[l]) {
          assert(!found);
          found = true;
        }
      }
      assert(found);
    }
  }
  fprintf(stderr, "%zu operators after fusion...\n", operators.size());
  for (size_t i = 0; i < operators.size(); i++) {
    Op *op = operators[i];
    printf("operator[%zu]: type(%s) guid(%lu)\n",
           i,
           get_operator_type_name(operators[i]->op_type).c_str(),
           operators[i]->op_guid);
    for (int j = 0; j < op->numInputs; j++) {
      LogicalRegion handle = op->inputs[j]->region;
      printf("inputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numOutputs; j++) {
      LogicalRegion handle = op->outputs[j]->region;
      printf("outputs[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
    for (int j = 0; j < op->numWeights; j++) {
      LogicalRegion handle = op->weights[j]->region;
      printf("weights[%d] region(%d,%d,%d)\n",
             j,
             handle.get_index_space().get_id(),
             handle.get_field_space().get_id(),
             handle.get_tree_id());
    }
  }
}


void FFModel::perform_inplace_optimizations() {
  for (size_t l = 1; l < operators.size(); l++) {
    if (operators[l]->can_inplace_output()) {
      // Assume outputs[0] is inplace with inputs[0]
      assert(operators[l]->numOutputs == 1);
      if (operators[l]->inputs[0]->owner_op != NULL) {
        // int dim1 = operators[l]->outputs[0]->num_dims;
        // int dim2 = operators[l]->inputs[0]->num_dims;
        MachineView view1 = operators[l]->outputs[0]->machine_view.value();
        MachineView view2 = operators[l]->inputs[0]->machine_view.value();
        if (view1 == view2) {
          // Check no others also need operators[l]->inputs[0]
          bool found = false;
          for (size_t i = 0; i < operators.size(); i++) {
            if (i == l) {
              continue;
            }
            for (int j = 0; j < operators[i]->numInputs; j++) {
              if ((operators[i]->inputs[j]->owner_op ==
                   operators[l]->inputs[0]->owner_op) &&
                  (operators[i]->inputs[j]->owner_idx ==
                   operators[l]->inputs[0]->owner_idx)) {
                found = true;
              }
            }
          }
          if (!found) {
            // Perform inplace
            operators[l]->do_inplace_output();
          }
        }
      }
    }
  }
}
bool FFModel::apply_fusion(std::vector<Op *> const &operators,
                           std::vector<Op *> &new_operators) {
  // Context ctx = config.lg_ctx;
  // Runtime* runtime = config.lg_hlr;
  for (size_t l = 1; l < operators.size() - 1; l++) {
    // don't fuse input and weight operator since they don't involve any
    // forward/backward task launches
    if (operators[l]->op_type == OP_INPUT ||
        operators[l]->op_type == OP_WEIGHT) {
      continue;
    }
    // don't fuse parallel op since they have different parallel_is in
    // forward/backward
    if (operators[l]->is_parallel_op()) {
      continue;
    }
    size_t start = 0;
    {
      Op *opl = operators[l];
      for (int idx = 0; idx < opl->numInputs; idx++) {
        bool found = false;
        for (size_t i = 0; i < l; i++) {
          if (opl->inputs[idx]->owner_op == operators[i]) {
            assert(!found);
            found = true;
            if (i > start) {
              start = i;
            }
          }
        }
        assert(found || (opl->inputs[idx]->owner_op == NULL));
      }
    }
    for (size_t i = start; i < l; i++) {
      // Domain d1 =
      // runtime->get_index_space_domain(operators[l]->outputs[0]->parallel_is);
      // Domain d2 =
      // runtime->get_index_space_domain(operators[i]->outputs[0]->parallel_is);
      MachineView view1 = operators[l]->outputs[0]->machine_view.value();
      MachineView view2 = operators[i]->outputs[0]->machine_view.value();
      if (view1 == view2) {
        FusedOp *fused_op = nullptr;
        bool allocate_new_fused_op = false;
        if (operators[i]->op_type == OP_FUSED) {
          fused_op = (FusedOp *)operators[i];
        } else {
          //  cannot be an in-place operator
          if (operators[i]->has_inplace_output()) {
            continue;
          }
          // don't fuse input and weight operator since they don't involve any
          // forward/backward kernels
          if (operators[i]->op_type == OP_INPUT ||
              operators[i]->op_type == OP_WEIGHT) {
            continue;
          }
          // don't fuse parallel op since they have different parallel_is in
          // forward/backward
          if (operators[i]->is_parallel_op()) {
            continue;
          }
          fused_op = new FusedOp(*this, operators[i]);
          allocate_new_fused_op = true;
        }
        if (fused_op->add_operator(*this, operators[l])) {
          // Construct new operators
          new_operators.clear();
          for (size_t j = 0; j < i; j++) {
            new_operators.push_back(operators[j]);
          }
          new_operators.push_back(fused_op);
          for (size_t j = i + 1; j < operators.size(); j++) {
            if (j == l) {
              continue; // l and i are fused
            }
            Op *op = operators[j];
            // Update input tensors that belong to operator[l] or operator[i]
            for (int idx = 0; idx < op->numInputs; idx++) {
              if ((op->inputs[idx]->owner_op == operators[l]) ||
                  (op->inputs[idx]->owner_op == operators[i])) {
                int found = -1;
                for (int k = 0; k < fused_op->numOutputs; k++) {
                  if (fused_op->outputs[k]->region == op->inputs[idx]->region) {
                    assert(found == -1);
                    found = k;
                  }
                }
                assert(found >= 0);
                op->inputs[idx] = fused_op->outputs[found];
              }
            }
            // Insert op
            new_operators.push_back(op);
          }
          // We are exact one operator fewer than the original
          assert(new_operators.size() + 1 == operators.size());
          return true;
        } else {
          // TODO: delete fused_op to avoid memory leakage
          if (allocate_new_fused_op) {
            delete fused_op;
          }
          continue;
        }
      }
    }
  }
  return false;
}

}
