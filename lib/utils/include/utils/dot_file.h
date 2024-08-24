#ifndef _DOT_FILE_H
#define _DOT_FILE_H

#include "record_formatter.h"
#include <cassert>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T>
class DotFile {
private:
  size_t node_id = 0;
  size_t subgraph_id = 0;
  std::map<T, size_t> node_ids;
  std::unordered_map<size_t, std::unordered_set<size_t>> subgraphs;
  std::unordered_map<size_t, std::unordered_set<size_t>> subgraph_children;
  std::unordered_map<size_t, std::optional<size_t>> subgraph_parents;
  std::optional<std::ofstream> owned_fstream = std::nullopt;
  std::ostream *out = nullptr;
  std::string get_node_name(size_t node_id) const {
    std::ostringstream s;
    s << "node" << node_id;
    return s.str();
  }
  bool has_ostream() const {
    return this->owned_fstream.has_value() || this->out != nullptr;
  }
  std::ostream &get_ostream() {
    bool has_owned_stream = this->owned_fstream.has_value();
    bool has_stream_ref = (this->out != nullptr);
    assert(has_owned_stream != has_stream_ref);
    if (has_owned_stream) {
      return this->owned_fstream.value();
    } else if (has_stream_ref) {
      return *this->out;
    } else {
      throw std::runtime_error("No ostream value set");
    }
  }
  void start_output() {
    this->get_ostream() << "digraph taskgraph {" << std::endl;
  }

public:
  DotFile() {}
  DotFile(std::string const &filename) : owned_fstream(filename) {
    this->start_output();
  }
  DotFile(std::ostream &s) : node_id(0), out(&s) {
    this->start_output();
  }

  void set_filename(std::string filename) {
    this->owned_fstream = std::ofstream(filename);
    this->start_output();
  }
  void reserve_node(T const &t) {
    if (this->node_ids.find(t) == this->node_ids.end()) {
      this->node_ids[t] = this->node_id++;
    }
  }
  void add_node(T const &t, std::map<std::string, std::string> const &params) {
    this->reserve_node(t);
    this->get_ostream() << "  " << this->get_node_name(this->node_ids.at(t))
                        << " [";
    for (auto it = params.begin(); it != params.end(); ++it) {
      this->get_ostream() << it->first << "=" << it->second;
      if (std::next(it) != params.end()) {
        this->get_ostream() << ",";
      }
    }
    this->get_ostream() << "];" << std::endl;
  }
  void add_record_node(T const &t, RecordFormatter const &rf) {
    std::ostringstream oss;
    oss << "\"" << rf << "\"";
    this->add_node(t, {{"shape", "record"}, {"label", oss.str()}});
  }

  void dump_subgraph(size_t subgraph) {
    this->get_ostream() << "subgraph cluster_" << subgraph << " {" << std::endl;
    for (size_t node_id : this->subgraphs.at(subgraph)) {
      this->get_ostream() << "node" << node_id << ";" << std::endl;
    }
    for (size_t child_subgraph : this->subgraph_children.at(subgraph)) {
      dump_subgraph(child_subgraph);
    }
    this->get_ostream() << "}" << std::endl;
  }

  void add_edge(T const &src, 
                T const &dst,
                std::optional<std::string> const &src_field = std::nullopt,
                std::optional<std::string> const &dst_field = std::nullopt) {
    this->reserve_node(src);
    this->reserve_node(dst);

    auto get_field_suffix = [](std::optional<std::string> const &field) -> std::string {
      if (field.has_value()) {
        return (":" + field.value());
      } else {
        return "";
      }
    };

    std::string src_name = this->get_node_name(this->node_ids.at(src));

    std::string dst_name = this->get_node_name(this->node_ids.at(dst));

    this->get_ostream() << "  " << src_name << get_field_suffix(src_field)
                        << " -> " << dst_name << get_field_suffix(dst_field) 
                        << ";" << std::endl;
  }
  void close() {
    for (size_t subgraph = 0; subgraph < this->subgraph_id; subgraph++) {
      if (!this->subgraph_parents.at(subgraph).has_value()) {
        this->dump_subgraph(subgraph);
      }
    }

    this->get_ostream() << "}";
    this->get_ostream().flush();
  }

  size_t add_subgraph(std::optional<size_t> parent_id = std::nullopt) {
    size_t subgraph = this->subgraph_id;
    subgraph_id++;
    this->subgraph_children[subgraph];
    if (parent_id.has_value()) {
      this->subgraph_children.at(parent_id.value()).insert(subgraph);
    }
    this->subgraph_parents[subgraph] = parent_id;
    this->subgraphs[subgraph];
    return subgraph;
  }

  void add_node_to_subgraph(T const &node, size_t subgraph) {
    this->reserve_node(node);
    if (subgraph >= this->subgraph_id) {
      std::ostringstream oss;
      oss << "Invalid subgraph_id " << subgraph << " (should be less than "
          << this->subgraph_id << ")";
      throw std::runtime_error(oss.str());
    }
    this->subgraphs[subgraph].insert(this->node_ids.at(node));
    std::optional<size_t> parent = this->subgraph_parents.at(subgraph);
    if (parent.has_value()) {
      this->add_node_to_subgraph(node, parent.value());
    }
  }
};

#endif // _DOT_FILE_H
