#ifndef _FLEXFLOW_RECURSIVE_LOGGER_H
#define _FLEXFLOW_RECURSIVE_LOGGER_H

#include "legion/legion_utilities.h"

namespace FlexFlow {

class RecursiveLogger {
public:
  /* RecursiveLogger(LegionRuntime::Logger::Category const &); */
  RecursiveLogger(std::string const &category_name);

  Realm::LoggerMessage debug();
  Realm::LoggerMessage spew();
  void enter();
  void leave();
private:
  int depth = 0;

  LegionRuntime::Logger::Category logger;
};

}; // namespace FlexFlow 
#endif // _FLEXFLOW_RECURSIVE_LOGGER_H
