#ifndef _FLEXFLOW_KERNELS_DATATYPE_DISPATCH_H
#define _FLEXFLOW_KERNELS_DATATYPE_DISPATCH_H

#include "accessor.h"

namespace FlexFlow {

template <template <DataType> typename F, typename ...Args>
void dispatch(DataType dt, Args&&... args) {
  switch (dt) {
    case DT_FLOAT:
      F<DT_FLOAT>{}(std::forward<Args>(args)...);
    case DT_DOUBLE:
      F<DT_DOUBLE>{}(std::forward<Args>(args)...);
    case DT_INT32:
      F<DT_INT32>{}(std::forward<Args>(args)...);
    case DT_INT64:
      F<DT_INT64>{}(std::forward<Args>(args)...);
    case DT_BOOLEAN:
      F<DT_BOOLEAN>{}(std::forward<Args>(args)...);
    default:
      throw std::runtime_error("Unknown datatype" + get_data_type_name(dt));
  }
}

template <template <DataType, DataType> typename F>
struct DataTypeDispatch2 {
  template <DataType IT>
  struct InputType {

    template <DataType OT>
    struct OutputType {
      template <typename ...Args>
      void operator()(Args ...args) const {
        F<IT, OT>{}(std::forward<Args>(args)...);
      }
    };

    template <typename ...Args>
    void operator()(DataType output_type, Args... args) const { 
      dispatch<OutputType>(output_type, std::forward<Args>(args)...);
    }
  };
  
  template <typename ...Args>
  void operator()(DataType input_data_type, DataType output_data_type, Args... args) {
    dispatch<InputType>(input_data_type, output_data_type, std::forward<Args>(args)...);
  }
};

}

#endif
