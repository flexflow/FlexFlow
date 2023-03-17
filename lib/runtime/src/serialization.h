#ifndef _FLEXFLOW_RUNTIME_SERIALIZATION_H
#define _FLEXFLOW_RUNTIME_SERIALIZATION_H

#include "legion.h"
#include "legion/legion_utilities.h"
#include "compiler/compiler.h"
#include "utils/optional.h"

namespace FlexFlow {

bool needs_serialize(SearchSolution const &);
std::size_t serialize(Legion::Serializer const &, SearchSolution const &);
std::size_t deserialize(Legion::Deserializer const &, SearchSolution &);

bool needs_serialize();

class FFTaskArgs {
  template <typename T>
  void add_arg(T const &t) {
    this->arg_types.push_back(typeid(t));
    this->offsets.push_back(this->inner_size());

    if (!this->inner.has_value() && !needs_serialize(t)) {
      this->inner = TaskArgument(&t, sizeof(T));
    } else {
      if (!this->inner.has_value()) {
        this->inner = Legion::Serializer{};
      }
      serialize(this->get_serializer(), t);
    }
  }

  Legion::TaskArgument get() {
    if (!inner.has_value()) {
      return Legion::TaskArgument(nullptr, 0);
    } else if (this->is_task_arg()) {
      return this->get_task_arg();
    } else {
      Legion::Serializer const &sez = this->get_serializer();
      return Legion::TaskArgument(sez.get_buffer(), sez.get_used_bytes());
    }
  }

  template <typename T>
  T const *at(int idx, void *args) {
    assert (this->arg_types.at(idx) == typeid(T));
    std::size_t offset = this->offsets.at(idx);
    return args;
  }

  template <typename T>
  T const *at(void *args) {
    for (int i = 0; i < this->arg_types.size(); i++) {
      if (this->arg_types.at(i) == typeid(T)) {
        return this->at<T>(i, args);
      }
    }

    {
      std::ostringstream oss;
      oss << "Could not find arg of requested type " << typeid(T).name();
      throw std::runtime_error(oss.str());
    }
  }

private:
  bool is_serializer() const {
    return this->inner.has_value() && mpark::holds_alternative<Legion::Serializer>(this->inner.value());
  }

  Legion::Serializer &get_serializer() {
    return mpark::get<Legion::Serializer>(this->inner.value());
  }

  Legion::Serializer const &get_serializer() const {
    return mpark::get<Legion::Serializer>(this->inner.value());
  }

  bool is_task_arg() const {
    return this->inner.has_value() && mpark::holds_alternative<Legion::TaskArgument>(this->inner.value());
  }

  Legion::TaskArgument &get_task_arg() {
    return mpark::get<Legion::TaskArgument>(this->inner.value());
  }

  Legion::TaskArgument const &get_task_arg() const {
    return mpark::get<Legion::TaskArgument>(this->inner.value());
  }

  std::size_t inner_size() const {
    if (!this->inner.has_value()) {
      return 0;
    }

    if (this->is_serializer()) {
      return this->get_serializer().get_used_bytes();
    } else {
      return this->get_task_arg().get_size();
    }
  }

  optional<variant<Legion::TaskArgument, Legion::Serializer>> inner;
  std::vector<std::type_info> arg_types;
  std::vector<std::size_t> offsets;
};

/* void deserialize_graph_optimal_view( */
/*     Legion::Deserializer &dez, */
/*     PCG::Graph *graph, */
/*     std::unordered_map<PCG::Node, MachineView> &optimal_views); */
}

#endif
