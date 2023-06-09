function(alias_library alias_name real_name)
  # https://cmake.org/cmake/help/v3.15/manual/cmake-buildsystem.7.html#alias-targets
  get_target_property(_aliased "${real_name}" ALIASED_TARGET)
  if(_aliased)
    add_library("${alias_name}" ALIAS "${_aliased}")
  else()
    add_library("${alias_name}" ALIAS "${real_name}")
  endif()
endfunction()
