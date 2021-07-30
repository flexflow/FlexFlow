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
  tl::optional<std::ostream&> out = tl::nullopt;
  std::string get_node_name(size_t node_id) const {
    std::ostringstream s;
    s << "node" << node_id;
    return s.str();
  }
  bool has_ostream() const {
    return this->owned_fstream.has_value() || this->out.has_value();
  }
  std::ostream& get_ostream() {
    bool has_owned_stream = this->owned_fstream.has_value();
    bool has_stream_ref = this->out.has_value();
    assert (has_owned_stream != has_stream_ref);
    if (has_owned_stream) {
      return this->owned_fstream.value();
    } else if (has_stream_ref) {
      return this->out.value();
    } else {
      throw std::runtime_error("No ostream value set");
    }
  }
  void start_output() {
    this->get_ostream() << "digraph taskgraph {" << std::endl;
  }
public:
  DotFile() : node_id(0) {}
  DotFile(std::string const &filename) 
    : node_id(0), owned_fstream(filename) 
  { 
    this->start_output();
  }
  DotFile(std::ostream& s)
    : node_id(0), out(s)
  {
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
    this->get_ostream() << "  " << this->get_node_name(this->node_ids.at(t)) << " [";
    for (auto it = params.begin(); it != params.end(); ++it)  {
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
    this->get_ostream() << "  " << src_name << " -> " << dst_name << ";" << std::endl;
  }
  void close() {
    this->get_ostream() << "}";
    this->get_ostream().flush();
  }
};

#endif // _DOT_FILE_H
