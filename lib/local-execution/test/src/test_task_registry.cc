#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/local_cost_estimator.h"
#include "local-execution/ops/attention.h"
#include "local-execution/task_signature_impl.h"
#include "pcg/computation_graph_builder.h"
#include "utils/fmt/optional.h"
#include "utils/fmt/unordered_map.h"

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Task Registry") {
    TaskRegistry task_registry;

    layer_guid_t layer_guid = layer_guid_t{Node{0}};
    int embed_dim = 32;
    int num_heads = 10;
    ComputationGraphOpAttrs attrs =
        ComputationGraphOpAttrs{MultiHeadAttentionAttrs{
            /*embed_dim=*/embed_dim,
            /*num_heads=*/num_heads,
            /*kdim=*/embed_dim,
            /*vdim=*/embed_dim,
            /*dropout=*/0.0,
            /*bias=*/true,
            /*add_bias_kv=*/false,
            /*add_zero_attn=*/false,
        }};

    SUBCASE("register single layer") {
      task_registry.register_tasks_for_layer(layer_guid, attrs);

      TaskRegistry correct_task_registry = [&] {
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            init_task_ids = {{layer_guid, task_id_t::ATTENTION_INIT_TASK_ID}};
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            fwd_task_ids = {{layer_guid, task_id_t::ATTENTION_FWD_TASK_ID}};
        std::unordered_map<layer_guid_t, std::optional<task_id_t>>
            bwd_task_ids = {{layer_guid, task_id_t::ATTENTION_BWD_TASK_ID}};
        std::unordered_map<task_id_t, TaskSignatureAndImpl> task_mapping = {
            {task_id_t::ATTENTION_INIT_TASK_ID,
             get_task_sig_impl(task_id_t::ATTENTION_INIT_TASK_ID)},
            {task_id_t::ATTENTION_FWD_TASK_ID,
             get_task_sig_impl(task_id_t::ATTENTION_FWD_TASK_ID)},
            {task_id_t::ATTENTION_BWD_TASK_ID,
             get_task_sig_impl(task_id_t::ATTENTION_BWD_TASK_ID)}};
        TaskRegistry correct;
        correct.init_task_ids = init_task_ids;
        correct.forward_task_ids = fwd_task_ids;
        correct.backward_task_ids = bwd_task_ids;
        correct.task_mapping = task_mapping;
        return correct;
      }();

      CHECK(task_registry == correct_task_registry);
    }

    SUBCASE("multiple layers same task") {
      layer_guid_t other_layer_guid = layer_guid_t{Node{1}};
      task_registry.register_tasks_for_layer(layer_guid, attrs);
      task_registry.register_tasks_for_layer(other_layer_guid, attrs);

      SUBCASE("layer to task ids") {
        std::unordered_map<layer_guid_t, std::optional<task_id_t>> correct = {
            {layer_guid, task_id_t::ATTENTION_INIT_TASK_ID},
            {other_layer_guid, task_id_t::ATTENTION_INIT_TASK_ID},
        };
        CHECK(correct == task_registry.init_task_ids);
      }

      std::unordered_map<task_id_t, TaskSignatureAndImpl> correct_task_mapping =
          {{task_id_t::ATTENTION_INIT_TASK_ID,
            get_task_sig_impl(task_id_t::ATTENTION_INIT_TASK_ID)},
           {task_id_t::ATTENTION_FWD_TASK_ID,
            get_task_sig_impl(task_id_t::ATTENTION_FWD_TASK_ID)},
           {task_id_t::ATTENTION_BWD_TASK_ID,
            get_task_sig_impl(task_id_t::ATTENTION_BWD_TASK_ID)}};
      SUBCASE("task to signature+impl mapping") {
        CHECK(correct_task_mapping == task_registry.task_mapping);
      }
      SUBCASE("different attrs, still same task fn mapping") {
        int embed_dim = 100;
        layer_guid_t layer_3 = layer_guid_t{Node{3}};
        ComputationGraphOpAttrs other_attrs =
            ComputationGraphOpAttrs{MultiHeadAttentionAttrs{
                /*embed_dim=*/embed_dim,
                /*num_heads=*/num_heads,
                /*kdim=*/embed_dim,
                /*vdim=*/embed_dim,
                /*dropout=*/0.0,
                /*bias=*/true,
                /*add_bias_kv=*/false,
                /*add_zero_attn=*/false,
            }};
        task_registry.register_tasks_for_layer(layer_3, other_attrs);

        CHECK(correct_task_mapping == task_registry.task_mapping);
      }
    }

    SUBCASE("equality") {
      TaskRegistry other_task_registry;
      SUBCASE("different attrs is still equal") {
        int embed_dim = 100;
        ComputationGraphOpAttrs other_attrs =
            ComputationGraphOpAttrs{MultiHeadAttentionAttrs{
                /*embed_dim=*/embed_dim,
                /*num_heads=*/num_heads,
                /*kdim=*/embed_dim,
                /*vdim=*/embed_dim,
                /*dropout=*/0.0,
                /*bias=*/true,
                /*add_bias_kv=*/false,
                /*add_zero_attn=*/false,
            }};

        task_registry.register_tasks_for_layer(layer_guid, attrs);
        other_task_registry.register_tasks_for_layer(layer_guid, other_attrs);

        CHECK(task_registry == other_task_registry);
      }

      SUBCASE("different layer_guid is not equal") {
        task_registry.register_tasks_for_layer(layer_guid, attrs);
        layer_guid_t other_layer_guid = layer_guid_t{Node{1}};
        other_task_registry.register_tasks_for_layer(other_layer_guid, attrs);

        CHECK(task_registry != other_task_registry);
      }
    }
  }
}

} // namespace FlexFlow
