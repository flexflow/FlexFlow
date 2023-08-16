# MPark.Variant
#
# Copyright Michael Park, 2015-2017
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

# Config file for MPark.Variant
#
#   `MPARK_VARIANT_INCLUDE_DIRS` - include directories
#   `MPARK_VARIANT_LIBRARIES`    - libraries to link against
#
# The following `IMPORTED` target is also defined:
#
#   `mpark_variant`


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was mpark_variant-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/mpark_variant-targets.cmake")

get_target_property(
  MPARK_VARIANT_INCLUDE_DIRS
  mpark_variant INTERFACE_INCLUDE_DIRECTORIES)

set_and_check(MPARK_VARIANT_INCLUDE_DIRS "${MPARK_VARIANT_INCLUDE_DIRS}")
set(MPARK_VARIANT_LIBRARIES mpark_variant)
