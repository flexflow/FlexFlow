#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"

#include "local-execution/local_cost_estimator.h"
#include "local-execution/ops/attention.h"
#include "local-execution/task_signature_impl.h"
#include "pcg/computation_graph_builder.h"

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
      SUBCASE("Task ids") {
        SUBCASE("Init task") {
          std::unordered_map<layer_guid_t, std::optional<task_id_t>> correct = {
              {layer_guid, ATTENTION_INIT_TASK_ID}};
          CHECK(correct == task_registry.init_task_ids);
        }
        SUBCASE("Fwd task") {
          std::unordered_map<layer_guid_t, std::optional<task_id_t>> correct = {
              {layer_guid, ATTENTION_FWD_TASK_ID}};
          CHECK(correct == task_registry.forward_task_ids);
        }
        SUBCASE("Bwd task") {
          std::unordered_map<layer_guid_t, std::optional<task_id_t>> correct = {
              {layer_guid, ATTENTION_BWD_TASK_ID}};
          CHECK(correct == task_registry.backward_task_ids);
        }
      }

      SUBCASE("Task Mapping") {
        std::unordered_map<task_id_t, TaskSignatureAndImpl>
            correct_task_mapping = {{ATTENTION_INIT_TASK_ID,
                                     get_task_sig_impl(ATTENTION_INIT_TASK_ID)},
                                    {ATTENTION_FWD_TASK_ID,
                                     get_task_sig_impl(ATTENTION_FWD_TASK_ID)},
                                    {ATTENTION_BWD_TASK_ID,
                                     get_task_sig_impl(ATTENTION_BWD_TASK_ID)}};
        CHECK(correct_task_mapping == task_registry.task_mapping);
      }
    }

    SUBCASE("multiple layers same task") {
      layer_guid_t layer_1 = layer_guid_t{Node{1}};
      layer_guid_t layer_2 = layer_guid_t{Node{2}};
      task_registry.register_tasks_for_layer(layer_guid, attrs);
      task_registry.register_tasks_for_layer(layer_1, attrs);
      task_registry.register_tasks_for_layer(layer_2, attrs);

      SUBCASE("layer to task ids") {
        std::unordered_map<layer_guid_t, std::optional<task_id_t>> correct = {
            {layer_guid, ATTENTION_INIT_TASK_ID},
            {layer_1, ATTENTION_INIT_TASK_ID},
            {layer_2, ATTENTION_INIT_TASK_ID},
        };
        CHECK(correct == task_registry.init_task_ids);
      }

      std::unordered_map<task_id_t, TaskSignatureAndImpl> correct_task_mapping =
          {{ATTENTION_INIT_TASK_ID, get_task_sig_impl(ATTENTION_INIT_TASK_ID)},
           {ATTENTION_FWD_TASK_ID, get_task_sig_impl(ATTENTION_FWD_TASK_ID)},
           {ATTENTION_BWD_TASK_ID, get_task_sig_impl(ATTENTION_BWD_TASK_ID)}};
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
