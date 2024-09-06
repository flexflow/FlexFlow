if(FF_BUILD_MOE OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/mixture_of_experts)
endif()

if(FF_BUILD_SUBSTITUTION_TOOL)
  add_subdirectory(protobuf_to_json)
endif()

if(FF_BUILD_VISUALIZATION_TOOL)
  add_subdirectory(substitutions-to-dot)
endif()

if(FF_BUILD_SP_IZATION_BENCHMARKING)
  add_subdirectory(sp_ization_benchmarking)
endif()

if(FF_BUILD_ARG_PARSER)
  add_subdirectory(arg_parser)
endif()
