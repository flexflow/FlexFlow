#ifndef _FLEXFLOW_RUNTIME_FF_TASK_ARGS_H
#define _FLEXFLOW_RUNTIME_FF_TASK_ARGS_H

#include "legion.h"
#include <typeinfo>
#include "legion/legion_utilities.h"
#include "utils/variant.h"
#include "utils/optional.h"
#include "serialization.h"

namespace FlexFlow {

class FFTaskArgs {
  template <typename T>
  void add_arg(T const &t) {
    this->arg_types.push_back(typeid(t));
    this->offsets.push_back(0);

    ff_task_serialize(this->sez, t);
  }

  Legion::TaskArgument get() {
    return Legion::TaskArgument(this->sez.get_buffer(), this->sez.get_used_bytes());
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
  Legion::Serializer sez;
  std::vector<std::type_info> arg_types;
  std::vector<std::size_t> offsets;
};

}

#endif 
