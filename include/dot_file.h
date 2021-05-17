#ifndef _DOT_FILE_H
#define _DOT_FILE_H

#include <fstream>
#include <sstream>
#include "tl/optional.h"

class RecordFormatter {
  friend RecordFormatter &operator<<(RecordFormatter &r, std::string const &tok) {
    r.pieces.push_back(tok);

    return r;
  }

  friend RecordFormatter &operator<<(RecordFormatter &r, RecordFormatter const &sub_r) {
    std::ostringstream oss;
    oss << sub_r;
    r << oss.str();

    return r;
  }

  friend RecordFormatter &operator<<(RecordFormatter &r, std::ostringstream &oss) {
    r << oss.str();

    return r;
  }

  friend std::ostream &operator<<(std::ostream &s, RecordFormatter const &r) {
    s << "{ ";
    for (size_t i = 0; i < r.pieces.size(); i++) {
      s << r.pieces[i];
      if (i + 1 < r.pieces.size()) {
        s << " | ";
      }
    }
    s << " }";

    return s;
  }
private:
  std::vector<std::string> pieces;
};

template <typename T>
class DotFile {
private:
  size_t node_id;
  std::map<T,size_t> node_ids;
  tl::optional<std::ofstream> owned_fstream = tl::nullopt;
  std::ostream& out;
  std::string get_node_name(size_t node_id) const {
    std::ostringstream s;
    s << "node" << node_id;
    return s.str();
  }
public:
  DotFile() : node_id(0) {}
  DotFile(std::string const &filename) : node_id(0), owned_fstream(filename), out(this->owned_fstream) { 
  }
  DotFile(std::ostream& s)
    : node_id(0), out(s)
  {
    out << "digraph taskgraph {";
  }

  void set_filename(std::string filename) {
    this->out = std::unique_ptr<std::ostream>(new std::ofstream(filename));
    out << "digraph taskgraph {";
  }
  void reserve_node(T const &t) {
    if (this->node_ids.find(t) == this->node_ids.end()) {
      this->node_ids[t] = this->node_id++;
    }
  }
  void add_node(T const &t, std::map<std::string, std::string> const &params) {
    this->reserve_node(t);
    out << "  " << this->get_node_name(this->node_ids.at(t)) << " [";
    for (auto it = params.begin(); it != params.end(); ++it)  {
      out << it->first << "=" << it->second;
      if (std::next(it) != params.end()) {
        out << ",";
      }
    }
    out << "];" << std::endl;
  }
  void add_record_node(T const &t, RecordFormatter const &rf) {
    std::ostringstream oss;
    oss << "\"" << rf << "\"";
    this->add_node(t,
      {
        {"shape", "record"},
        {"label", oss.str()}
      }
    );
  }
  void add_edge(T const &src, T const &dst) {
    this->reserve_node(src);
    this->reserve_node(dst);
    auto src_name = this->get_node_name(this->node_ids.at(src));
    auto dst_name = this->get_node_name(this->node_ids.at(dst));
    out << "  " << src_name << " -> " << dst_name << ";" << std::endl;
  }
  void close() {
    out << "}";
    out.flush();
  }
};

#endif // _DOT_FILE_H
