#ifndef _FLEXFLOW_UTILS_NOT_IMPLEMENTED_EXCEPTION_H
#define _FLEXFLOW_UTILS_NOT_IMPLEMENTED_EXCEPTION_H

#include <stdexcept>

class NotImplemented : public std::logic_error {
public:
  NotImplemented() : std::logic_error("Function not yet implemented"){};
};

#endif // _FLEXFLOW_UTILS_NOT_IMPLEMENTED_EXCEPTION_H
