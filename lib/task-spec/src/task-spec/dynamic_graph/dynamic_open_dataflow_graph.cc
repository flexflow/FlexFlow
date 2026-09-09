#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "op-attrs/ff_ordered/ff_ordered_filtrans.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "task-spec/dynamic_graph/dynamic_graph_edge.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_attrs.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/bidict/algorithms/unordered_bidict_from_map.h"
#include "utils/bidict/unordered_bidict.h"
#include "utils/containers/all_of.h"
#include "utils/containers/at_idx.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/contains_duplicates.h"
#include "utils/containers/contains_value.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/invert_map.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/multiset_of.h"
#include "utils/containers/multiset_union.h"
#include "utils/containers/repeat.h"
#include "utils/containers/require_all_of.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip_strict.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/find_isomorphism_between_labelled_open_kwarg_dataflow_graphs.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_graph_input.dtg.h"
#include "utils/many_to_one/many_to_one.h"

namespace FlexFlow {

DynamicOpenDataflowGraph make_empty_dynamic_open_dataflow_graph() {
  return DynamicOpenDataflowGraph{
      std::set<DynamicNodeInvocation>{},
      unordered_bidict<dynamic_invocation_id_t, DynamicNodeInvocation>{},
      unordered_bidict<dynamic_value_id_t, DynamicValueAttrs>{},
  };
}

void check_dynamic_open_dataflow_graph_is_valid(
    DynamicOpenDataflowGraph const &g) {
  std::map<DynamicValueAttrs, std::vector<DynamicNodeInvocation>>
      invocations_by_value_produced;

  for (DynamicNodeInvocation const &i : g.invocations) {
    for (DynamicValueAttrs const &v : values(i.outputs)) {
      invocations_by_value_produced[v].push_back(i);
    }
  }

  std::map<DynamicValueAttrs, std::vector<DynamicNodeInvocation>>
      values_produced_multiple_times = filter_values(
          invocations_by_value_produced,
          [](std::vector<DynamicNodeInvocation> const &producers) -> bool {
            return producers.size() > 1;
          });

  ASSERT(values_produced_multiple_times.size() == 0,
         keys(values_produced_multiple_times));

  // since DynamicOpenDataflowGraph contains a set of invocations rather than a
  // LabelledOpenKwargDataflowGraph, some properties guaranteed by
  // LabelledOpenKwargDataflowGraph (e.g., the graph is acyclic, all tensors
  // originate from another operator's output unless they're a designated graph
  // input, etc.) are not automatically guaranteed. Since
  // LabelledOpenKwargDataflowGraph guarantees these properties, the easiest
  // way to check them is to try to convert the DynamicOpenDataflowGraph into a
  // LabelledOpenKwargDataflowGraph, and if a value is returned without an
  // assertion we know the properties hold.
  labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(g);
}

DynamicOpenDataflowGraph compute_invocation_ids_for_dynamic_open_dataflow_graph(
    DynamicOpenDataflowGraph const &g) {
  unordered_bidict<dynamic_invocation_id_t, DynamicNodeInvocation>
      invocation_ids = unordered_bidict_from_map(
          map_keys(enumerate(g.invocations), [](nonnegative_int i) {
            return dynamic_invocation_id_t{i};
          }));

  DynamicOpenDataflowGraph result{
      /*invocations=*/g.invocations,
      /*invocation_ids=*/invocation_ids,
      /*value_ids=*/g.value_ids,
  };
  return result;
}

DynamicOpenDataflowGraph compute_value_ids_for_dynamic_open_dataflow_graph(
    DynamicOpenDataflowGraph const &g) {
  std::map<dynamic_value_id_t, DynamicValueAttrs> internal_value_ids = map_keys(
      enumerate(dynamic_graph_get_internal_values(g)), [](nonnegative_int i) {
        return dynamic_value_id_t{dynamic_internal_value_id_t{i}};
      });

  std::map<dynamic_value_id_t, DynamicValueAttrs> external_value_ids = map_keys(
      enumerate(dynamic_graph_get_external_values(g)), [](nonnegative_int i) {
        return dynamic_value_id_t{dynamic_external_value_id_t{i}};
      });

  unordered_bidict<dynamic_value_id_t, DynamicValueAttrs> value_ids =
      unordered_bidict_from_map(
          binary_merge_disjoint_maps(internal_value_ids, external_value_ids));

  DynamicOpenDataflowGraph result{
      /*invocations=*/g.invocations,
      /*invocation_ids=*/g.invocation_ids,
      /*value_ids=*/value_ids,
  };
  return result;
}

nonnegative_int dynamic_graph_num_nodes(DynamicOpenDataflowGraph const &g) {
  return num_elements(get_dynamic_nodes(g));
}

bool full_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &g,
    std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
    std::function<bool(DynamicValueAttrs const &)> const &value_condition,
    std::function<bool(DynamicTensorSlot const &)> const &slot_condition) {

  return all_of(get_dynamic_nodes(g), node_condition) &&
         all_of(get_dynamic_values(g), value_condition) &&
         all_of(get_dynamic_tensor_slots(g), slot_condition);
}

bool no_part_of_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &g,
    std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
    std::function<bool(DynamicValueAttrs const &)> const &value_condition,
    std::function<bool(DynamicTensorSlot const &)> const &slot_condition) {

  return full_dynamic_graph_satisfies(
      g,
      [&](DynamicNodeAttrs const &n) -> bool { return !node_condition(n); },
      [&](DynamicValueAttrs const &v) -> bool { return !value_condition(v); },
      [&](DynamicTensorSlot const &s) -> bool { return !slot_condition(s); });
}

void require_full_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &g,
    std::function<void(DynamicNodeInvocation const &)> const
        &invocation_condition) {
  require_all_of(g.invocations, invocation_condition);
}

std::multiset<DynamicNodeAttrs>
    get_dynamic_nodes(DynamicOpenDataflowGraph const &g) {
  return transform(multiset_of(g.invocations),
                   [&](DynamicNodeInvocation const &i) -> DynamicNodeAttrs {
                     return i.node_attrs;
                   });
}

std::multiset<DynamicValueAttrs>
    get_dynamic_values(DynamicOpenDataflowGraph const &g) {
  return flatmap(
      multiset_of(g.invocations),
      [&](DynamicNodeInvocation const &i) -> std::multiset<DynamicValueAttrs> {
        return multiset_union(values(i.inputs), values(i.outputs));
      });
}

std::multiset<DynamicTensorSlot>
    get_dynamic_tensor_slots(DynamicOpenDataflowGraph const &g) {
  return flatmap(
      multiset_of(g.invocations),
      [&](DynamicNodeInvocation const &i) -> std::multiset<DynamicTensorSlot> {
        return multiset_of(set_union(keys(i.inputs), keys(i.outputs)));
      });
}

std::set<DynamicNodeInvocation>
    get_dynamic_invocation_set(DynamicOpenDataflowGraph const &g) {
  return g.invocations;
}

std::set<DynamicValueAttrs>
    dynamic_graph_get_internal_values(DynamicOpenDataflowGraph const &g) {
  std::set<InternalDynamicSlotSite> internal_slot_sites =
      get_internal_dynamic_slot_sites(g);

  std::set<DynamicValueAttrs> internal_values = filtrans(
      internal_slot_sites,
      [&](InternalDynamicSlotSite const &s)
          -> std::optional<DynamicValueAttrs> {
        if (s.direction == TensorDirection::OUTPUT) {
          return dynamic_value_attrs_for_slot_site(g, DynamicSlotSite{s});
        } else {
          return std::nullopt;
        }
      });

  return internal_values;
}

std::set<DynamicValueAttrs>
    dynamic_graph_get_external_values(DynamicOpenDataflowGraph const &g) {
  std::set<DynamicValueAttrs> all_values = set_of(get_dynamic_values(g));

  std::set<DynamicValueAttrs> internal_values =
      dynamic_graph_get_internal_values(g);

  return set_minus(all_values, internal_values);
}

dynamic_invocation_id_t dynamic_graph_get_id_for_invocation(
    DynamicOpenDataflowGraph const &g,
    DynamicNodeInvocation const &invocation) {
  return g.invocation_ids.at_r(invocation);
}

DynamicNodeInvocation
    dynamic_graph_get_invocation_for_id(DynamicOpenDataflowGraph const &g,
                                        dynamic_invocation_id_t const &id) {
  return g.invocation_ids.at_l(id);
}

dynamic_value_id_t
    dynamic_graph_get_id_for_value(DynamicOpenDataflowGraph const &g,
                                   DynamicValueAttrs const &value) {
  return g.value_ids.at_r(value);
}

DynamicValueAttrs
    dynamic_graph_get_value_for_id(DynamicOpenDataflowGraph const &g,
                                   dynamic_value_id_t const &id) {
  return g.value_ids.at_l(id);
}

std::set<DynamicGraphEdge>
    get_dynamic_graph_edges(DynamicOpenDataflowGraph const &g) {
  return flatmap(
      get_dynamic_invocation_set(g),
      [&](DynamicNodeInvocation const &i) -> std::set<DynamicGraphEdge> {
        return get_dynamic_graph_edges_incoming_to_invocation(g, i);
      });
}

std::set<DynamicGraphEdge> get_dynamic_graph_edges_incoming_to_invocation(
    DynamicOpenDataflowGraph const &g, DynamicNodeInvocation const &i) {
  return transform(
      set_of(i.inputs),
      [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
          -> DynamicGraphEdge {
        DynamicSlotSite src = dynamic_graph_find_source_of_value(g, p.second);

        InternalDynamicSlotSite dst = InternalDynamicSlotSite{
            /*invocation_id=*/dynamic_graph_get_id_for_invocation(g, i),
            /*direction=*/TensorDirection::INCOMING,
            /*slot_name=*/p.first,
        };

        return dynamic_graph_edge_from_slot_sites(src, dst);
      });
}

std::set<DynamicGraphEdge> get_dynamic_graph_edges_outgoing_from_invocation(
    DynamicOpenDataflowGraph const &g, DynamicNodeInvocation const &i) {
  return flatmap(
      set_of(i.outputs),
      [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
          -> std::set<DynamicGraphEdge> {
        DynamicSlotSite src = DynamicSlotSite{
            InternalDynamicSlotSite{
                /*invocation_id=*/dynamic_graph_get_id_for_invocation(g, i),
                /*direction=*/TensorDirection::OUTPUT,
                /*slot_name=*/p.first,
            },
        };

        return transform(
            dynamic_graph_find_sinks_of_value(g, p.second),
            [&](InternalDynamicSlotSite const &sink) -> DynamicGraphEdge {
              return dynamic_graph_edge_from_slot_sites(src, sink);
            });
      });
}

DynamicValueAttrs
    dynamic_value_attrs_for_slot_site(DynamicOpenDataflowGraph const &g,
                                      DynamicSlotSite const &slot) {
  return slot.visit<DynamicValueAttrs>(overload{

      [&](ExternalDynamicSlotSite const &external_slot) -> DynamicValueAttrs {
        dynamic_value_id_t value_id =
            dynamic_value_id_t{external_slot.value_id};

        return dynamic_graph_get_value_for_id(g, value_id);
      },

      [&](InternalDynamicSlotSite const &internal_slot) -> DynamicValueAttrs {
        DynamicNodeInvocation invocation =
            dynamic_graph_get_invocation_for_id(g, internal_slot.invocation_id);
        switch (internal_slot.direction) {
          case TensorDirection::INCOMING:
            return invocation.inputs.at(internal_slot.slot_name);
          case TensorDirection::OUTPUT:
            return invocation.outputs.at(internal_slot.slot_name);
          default:
            PANIC("Unexpected direction {}", internal_slot.direction);
        }
      }});
}

std::set<InternalDynamicSlotSite>
    get_internal_dynamic_slot_sites(DynamicOpenDataflowGraph const &g) {
  return flatmap(
      get_dynamic_invocation_set(g),
      [&](DynamicNodeInvocation const &i) -> std::set<InternalDynamicSlotSite> {
        dynamic_invocation_id_t id = dynamic_graph_get_id_for_invocation(g, i);

        return get_dynamic_slot_sites_for_invocation(id, i);
      });
}

std::set<DynamicSlotSite>
    get_dynamic_slot_sites(DynamicOpenDataflowGraph const &g) {

  std::set<InternalDynamicSlotSite> internal_slot_sites =
      get_internal_dynamic_slot_sites(g);

  std::set<DynamicValueAttrs> external_values =
      dynamic_graph_get_external_values(g);

  std::set<ExternalDynamicSlotSite> external_slot_sites = transform(
      external_values,
      [&](DynamicValueAttrs const &external_value) -> ExternalDynamicSlotSite {
        dynamic_external_value_id_t value_id =
            dynamic_graph_get_id_for_value(g, external_value)
                .require_external();
        return ExternalDynamicSlotSite{value_id};
      });

  return set_union(
      transform(internal_slot_sites,
                [](InternalDynamicSlotSite const &s) -> DynamicSlotSite {
                  return DynamicSlotSite{s};
                }),
      transform(external_slot_sites,
                [](ExternalDynamicSlotSite const &s) -> DynamicSlotSite {
                  return DynamicSlotSite{s};
                }));
}

std::set<InternalDynamicSlotSite>
    dynamic_graph_find_sinks_of_value(DynamicOpenDataflowGraph const &g,
                                      DynamicValueAttrs const &v) {
  std::set<InternalDynamicSlotSite> found = filter(
      get_internal_dynamic_slot_sites(g),
      [&](InternalDynamicSlotSite const &s) -> bool {
        return dynamic_value_attrs_for_slot_site(g, DynamicSlotSite{s}) == v &&
               s.direction == TensorDirection::INCOMING;
      });

  return found;
}

DynamicSlotSite dynamic_graph_find_source_of_slot_site(
    DynamicOpenDataflowGraph const &g,
    InternalDynamicSlotSite const &slot_site) {
  DynamicValueAttrs value_attrs =
      dynamic_value_attrs_for_slot_site(g, DynamicSlotSite{slot_site});
  DynamicSlotSite src_site = dynamic_graph_find_source_of_value(g, value_attrs);
  return src_site;
}

std::set<InternalDynamicSlotSite> dynamic_graph_find_sinks_of_slot_site(
    DynamicOpenDataflowGraph const &g,
    InternalDynamicSlotSite const &slot_site) {
  DynamicValueAttrs value_attrs =
      dynamic_value_attrs_for_slot_site(g, DynamicSlotSite{slot_site});
  std::set<InternalDynamicSlotSite> sink_sites =
      dynamic_graph_find_sinks_of_value(g, value_attrs);
  return sink_sites;
}

DynamicSlotSite
    dynamic_graph_find_source_of_value(DynamicOpenDataflowGraph const &g,
                                       DynamicValueAttrs const &v) {

  dynamic_value_id_t value_id = dynamic_graph_get_id_for_value(g, v);

  auto is_source_of_value = [&](DynamicSlotSite const &s) -> bool {
    return s.visit<bool>(overload{
        [&](InternalDynamicSlotSite const &internal_slot_site) -> bool {
          return dynamic_value_attrs_for_slot_site(g, s) == v &&
                 internal_slot_site.direction == TensorDirection::OUTPUT;
        },
        [&](ExternalDynamicSlotSite const &external_slot_site) -> bool {
          return dynamic_value_id_t{external_slot_site.value_id} == value_id;
        },
    });
  };

  std::set<DynamicSlotSite> found =
      filter(get_dynamic_slot_sites(g), is_source_of_value);

  return get_only(found);
}

std::optional<DynamicValueAttrs>
    find_output_value_attrs(DynamicOpenDataflowGraph const &dg,
                            dynamic_tensor_guid_t tensor_guid,
                            std::optional<DynamicTensorRole> const &role) {
  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    for (auto const &[slot, output] : invocation.outputs) {
      if (output.tensor_guid == tensor_guid && output.role == role) {
        return output;
      }
    }
  }
  return std::nullopt;
}

DynamicOpenDataflowGraph transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &g,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const
        &f) {
  std::set<DynamicNodeInvocation> current_invocation_set =
      get_dynamic_invocation_set(g);
  std::set<DynamicNodeInvocation> new_invocation_set =
      transform(current_invocation_set, f);

  return dynamic_open_dataflow_graph_from_invocation_set(new_invocation_set);
}

DynamicOpenDataflowGraph flatmap_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &g,
    std::function<std::set<DynamicNodeInvocation>(
        DynamicNodeInvocation const &)> const &f) {

  std::set<DynamicNodeInvocation> current_invocation_set =
      get_dynamic_invocation_set(g);
  std::vector<DynamicNodeInvocation> new_invocation_set =
      flatmap(vector_of(current_invocation_set), f);

  ASSERT(!contains_duplicates(new_invocation_set));

  return dynamic_open_dataflow_graph_from_invocation_set(
      set_of(new_invocation_set));
}

DynamicOpenDataflowGraph dynamic_open_dataflow_graph_from_invocation_set(
    std::set<DynamicNodeInvocation> const &invocation_set) {

  DynamicOpenDataflowGraph result = DynamicOpenDataflowGraph{
      invocation_set,
      unordered_bidict<dynamic_invocation_id_t, DynamicNodeInvocation>{},
      unordered_bidict<dynamic_value_id_t, DynamicValueAttrs>{},
  };

  check_dynamic_open_dataflow_graph_is_valid(result);

  // note: invocation ids must be computed before value ids, as computing
  // value ids requires looking up invocation ids
  result = compute_invocation_ids_for_dynamic_open_dataflow_graph(result);

  return compute_value_ids_for_dynamic_open_dataflow_graph(result);
}

std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                         DynamicValueAttrs,
                                         int,
                                         DynamicTensorSlot>,
          bidict<Node, DynamicNodeInvocation>>
    labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
        DynamicOpenDataflowGraph const &g) {

  std::set<DynamicValueAttrs> all_values = set_of(get_dynamic_values(g));

  ManyToOne<DynamicValueAttrs, DynamicNodeInvocation> value_to_producer;
  for (DynamicNodeInvocation const &invocation :
       get_dynamic_invocation_set(g)) {
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      value_to_producer.insert({output, invocation});
    }
  }

  std::set<DynamicValueAttrs> graph_inputs =
      filter(all_values, [&](DynamicValueAttrs const &v) -> bool {
        return !value_to_producer.contains_l(v);
      });

  LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                 DynamicValueAttrs,
                                 int,
                                 DynamicTensorSlot>
      result = LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                              DynamicValueAttrs,
                                              int,
                                              DynamicTensorSlot>::
          create<
              UnorderedSetLabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                                         DynamicValueAttrs,
                                                         int,
                                                         DynamicTensorSlot>>();

  // note: unordered so that the contains_r probe in inputs_have_been_added
  // below is a hash lookup rather than a tree descent using
  // DynamicValueAttrs::operator<
  unordered_bidict<OpenKwargDataflowValue<int, DynamicTensorSlot>,
                   DynamicValueAttrs>
      value_map;

  for (auto const &kv : enumerate(graph_inputs)) {
    int input_idx = kv.first.unwrap_nonnegative();
    DynamicValueAttrs graph_input = kv.second;
    KwargDataflowGraphInput<int> added =
        result.add_input(input_idx, graph_input);
    value_map.equate(OpenKwargDataflowValue<int, DynamicTensorSlot>{added},
                     graph_input);
  }

  bidict<Node, DynamicNodeInvocation> node_map;
  std::set<DynamicNodeInvocation> to_add = g.invocations;

  auto add_invocation_to_graph =
      [&](DynamicNodeInvocation const &invocation) -> void {
    KwargNodeAddedResult<DynamicTensorSlot> added = result.add_node(
        invocation.node_attrs,
        map_values(invocation.inputs,
                   [&](DynamicValueAttrs const &input)
                       -> OpenKwargDataflowValue<int, DynamicTensorSlot> {
                     return value_map.at_r(input);
                   }),
        invocation.outputs);
    node_map.equate(added.node, invocation);

    for (auto const &[k, v] :
         zip_values_strict(invocation.outputs, added.outputs)) {
      DynamicValueAttrs invocation_output = v.first;
      KwargDataflowOutput<DynamicTensorSlot> graph_output = v.second;
      value_map.equate(
          OpenKwargDataflowValue<int, DynamicTensorSlot>{graph_output},
          invocation_output);
    }

    to_add.erase(invocation);
  };

  // Add invocations in topological order using a worklist of invocations
  // whose inputs are all available, rather than rescanning every remaining
  // invocation after each insertion (which is quadratic in the number of
  // invocations).
  //
  // Invocations are indexed by their position in g.invocations, which is
  // ordered, and the worklist is drained lowest-index-first. This reproduces
  // the selection made by a scan that takes the first ready invocation in
  // that same order.
  std::vector<DynamicNodeInvocation> invocation_by_idx =
      vector_of(g.invocations);
  int num_invocations = static_cast<int>(invocation_by_idx.size());

  std::unordered_map<DynamicValueAttrs, std::vector<int>> consumers_of_value;
  std::vector<int> num_unavailable_inputs(num_invocations, 0);

  for (int idx = 0; idx < num_invocations; idx++) {
    // a value consumed by several slots of one invocation only blocks it once
    for (DynamicValueAttrs const &input :
         set_of(values(invocation_by_idx.at(idx).inputs))) {
      if (!value_map.contains_r(input)) {
        num_unavailable_inputs.at(idx)++;
        consumers_of_value[input].push_back(idx);
      }
    }
  }

  std::set<int> ready;
  for (int idx = 0; idx < num_invocations; idx++) {
    if (num_unavailable_inputs.at(idx) == 0) {
      ready.insert(idx);
    }
  }

  while (!ready.empty()) {
    int idx = *ready.begin();
    ready.erase(ready.begin());

    DynamicNodeInvocation const &invocation = invocation_by_idx.at(idx);
    add_invocation_to_graph(invocation);

    for (DynamicValueAttrs const &output : set_of(values(invocation.outputs))) {
      for (int consumer_idx : consumers_of_value[output]) {
        if (--num_unavailable_inputs.at(consumer_idx) == 0) {
          ready.insert(consumer_idx);
        }
      }
    }
  }

  // any invocation still pending has an input that is never produced, or
  // participates in a cycle
  if (!to_add.empty()) {
    PANIC("Failed to add any invocations in to_add", to_add);
  }

  return std::pair{result, node_map};
}

bool dynamic_open_dataflow_graphs_are_isomorphic(
    DynamicOpenDataflowGraph const &lhs, DynamicOpenDataflowGraph const &rhs) {
  LabelledOpenKwargDataflowGraphView<DynamicNodeAttrs,
                                     DynamicValueAttrs,
                                     int,
                                     DynamicTensorSlot>
      lhs_dataflow_graph =
          labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
              lhs)
              .first;

  LabelledOpenKwargDataflowGraphView<DynamicNodeAttrs,
                                     DynamicValueAttrs,
                                     int,
                                     DynamicTensorSlot>
      rhs_dataflow_graph =
          labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
              rhs)
              .first;

  return find_isomorphism_between_labelled_open_kwarg_dataflow_graphs(
             lhs_dataflow_graph, rhs_dataflow_graph)
      .has_value();
}

std::string
    dynamic_open_dataflow_graph_as_dot(DynamicOpenDataflowGraph const &g) {
  std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                           DynamicValueAttrs,
                                           int,
                                           DynamicTensorSlot>,
            bidict<Node, DynamicNodeInvocation>>
      labelled_result =
          labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
              g);

  LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                 DynamicValueAttrs,
                                 int,
                                 DynamicTensorSlot>
      labelled_g = labelled_result.first;

  bidict<Node, DynamicNodeInvocation> invocations = labelled_result.second;

  auto dot_for_training_operation_attrs =
      [](TrainingOperationAttrs const &training_attrs) -> nlohmann::json {
    nlohmann::json result = training_attrs;

    return result;
  };

  std::function<nlohmann::json(DynamicNodeAttrs const &)> render_node_label =
      [](DynamicNodeAttrs const &a) -> nlohmann::json {
    nlohmann::json result = dynamic_node_attrs_to_serializable(a);

    return result;
  };

  auto render_parallel_tensor_space_coord =
      [](ParallelTensorSpaceCoordinate const &c) -> std::string {
    std::vector<std::string> replica_dim_entries = {
        fmt::format("+/{}", c.sum_component),
        fmt::format("=/{}", c.discard_copy_component),
    };

    std::vector<std::string> shard_entries = transform(
        vector_of(c.shard_components),
        [](nonnegative_int x) -> std::string { return fmt::to_string(x); });

    return (
        "(" +
        join_strings(concat_vectors(replica_dim_entries, shard_entries), ", ") +
        ")");
  };

  auto render_parallel_tensor_mapping =
      [](ParallelTensorMapping const &mapping) -> RecordFormatter {
    return mk_record_for_map(mapping.raw.as_map());
  };

  std::function<nlohmann::json(DynamicValueAttrs const &)> render_value_label =
      [&](DynamicValueAttrs const &a) -> nlohmann::json {
    nlohmann::json result = dynamic_value_attrs_to_serializable(a);
    return result;
  };

  std::function<nlohmann::json(DynamicTensorSlot const &)> render_slot_name =
      [](DynamicTensorSlot const &slot_name) -> nlohmann::json {
    nlohmann::json result = slot_name;
    return result;
  };

  std::function<std::vector<DynamicTensorSlot>(
      std::set<DynamicTensorSlot> const &)>
      order_slots = [](std::set<DynamicTensorSlot> const &slot_names)
      -> std::vector<DynamicTensorSlot> { return sorted(slot_names); };

  return labelled_open_kwarg_dataflow_graph_view_as_dot(labelled_g,
                                                        render_node_label,
                                                        render_value_label,
                                                        render_slot_name,
                                                        order_slots);
}

void debug_print_dynamic_open_dataflow_graph_as_dot(
    DynamicOpenDataflowGraph const &g) {
  std::cerr << dynamic_open_dataflow_graph_as_dot(g) << std::endl;
}

} // namespace FlexFlow
