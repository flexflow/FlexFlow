#ifndef _FLEXFLOW_RECURSIVE_LOGGER_H
#define _FLEXFLOW_RECURSIVE_LOGGER_H

#include "legion/legion_utilities.h"
#include <memory>

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define UNIQUE_TAG() CONCAT(tag, __COUNTER__)
#define TAG_ENTER(mlogger) auto UNIQUE_TAG() = mlogger->enter_tag()

namespace FlexFlow {

class RecursiveLogger;

class DepthTag {
public:
  DepthTag() = delete;
  DepthTag(RecursiveLogger &);
  DepthTag(DepthTag const &) = delete;
  ~DepthTag();

private:
  RecursiveLogger &logger;
};

class RecursiveLogger {
public:
  /* RecursiveLogger(Legion::Logger const &); */
  RecursiveLogger(std::string const &category_name);

  Realm::LoggerMessage info();
  Realm::LoggerMessage debug();
  Realm::LoggerMessage spew();
  void enter();
  void leave();

  std::unique_ptr<DepthTag> enter_tag();

private:
  int depth = 0;

  void print_prefix(Realm::LoggerMessage &) const;

  Legion::Logger logger;
};

};     // namespace FlexFlow
#endif // _FLEXFLOW_RECURSIVE_LOGGER_H
