#include "utils/record_formatter.h"

RecordFormatter &operator<<(RecordFormatter &r, std::string const &tok) {
  r.pieces.push_back(tok);

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, int tok) {
  std::ostringstream oss;
  oss << tok;

  r << oss;

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, float tok) {
  std::ostringstream oss;
  oss << std::scientific;
  oss << tok;

  r << oss;

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, RecordFormatter const &sub_r) {
  std::ostringstream oss;
  oss << sub_r;
  r << oss.str();

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, std::ostringstream &oss) {
  r << oss.str();

  return r;
}

std::ostream &operator<<(std::ostream &s, RecordFormatter const &r) {
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
