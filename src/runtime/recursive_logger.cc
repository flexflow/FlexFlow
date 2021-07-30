#include "flexflow/utils/recursive_logger.h"

namespace FlexFlow {

RecursiveLogger::RecursiveLogger(std::string const &category_name)
  : logger(category_name)
{ }

Realm::LoggerMessage RecursiveLogger::debug() {
  Realm::LoggerMessage msg = this->logger.debug();
  msg << this->depth << " ";
  for (int i = 0; i < this->depth; i++) {
    msg << " ";
  }

  return msg;
}

Realm::LoggerMessage RecursiveLogger::spew() {
  Realm::LoggerMessage msg = this->logger.spew();
  msg << this->depth << " ";
  for (int i = 0; i < this->depth; i++) {
    msg << " ";
  }

  return msg;
}

void RecursiveLogger::enter() {
  this->depth++;
}

void RecursiveLogger::leave() { 
  this->depth--;
}

}; // namespace FlexFlow
