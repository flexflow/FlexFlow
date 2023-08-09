#ifndef _RECORD_FORMATTER_H
#define _RECORD_FORMATTER_H

#include <sstream>
#include <vector>

class RecordFormatter {
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     std::string const &tok);
  friend RecordFormatter &operator<<(RecordFormatter &r, int tok);
  friend RecordFormatter &operator<<(RecordFormatter &r, float tok);
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     RecordFormatter const &sub_r);
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     std::ostringstream &oss);
  friend std::ostream &operator<<(std::ostream &s, RecordFormatter const &r);

private:
  std::vector<std::string> pieces;
};

#endif // _RECORD_FORMATTER_H
