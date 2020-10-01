#=============================================================================
# Copyright 2020 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# This module produces the "GASNet" link target which carries with it all the
# necessary interface properties.  The following options determine which
# GASNet backend configuration get's used:
#
# GASNet_CONDUIT   - Communication conduit to use
# GASNet_THREADING - Threading mode to use
#
# Valid options for these are dependenent on the specific GASNet installation
#
# GASNet_PREFERRED_CONDUITS - A list of conduits in order of preference to be
#                             used by default if found.  The default order of
#                             preference is mxm, psm, gemini, aries, pami, ibv,
#                             shmem, portals4, ofi, mpi, udp, and smp.
# GASNet_ROOT_DIR           - Prefix to use when searching for GASNet.  If
#                             specified then this search path will be used
#                             exclusively and all others ignored.
# ENV{GASNET_ROOT}          - Environment variable used to initialize the
#                             value of GASNet_ROOT_DIR if not already specified
#

macro(_GASNet_parse_conduit_and_threading_names
  MAKEFILE CONDUIT_LIST_VAR THREADING_LIST_VAR)
  get_filename_component(_BASE ${MAKEFILE} NAME_WE)
  string(REGEX MATCH "^([^\\-]*)-([^\\-]*)$" _BASE "${_BASE}")
  list(FIND ${CONDUIT_LIST_VAR} "${CMAKE_MATCH_1}" _I)
  if(_I EQUAL -1)
    list(APPEND ${CONDUIT_LIST_VAR} "${CMAKE_MATCH_1}")
  endif()
  list(FIND ${THREADING_LIST_VAR} "${CMAKE_MATCH_2}" _I)
  if(_I EQUAL -1)
    list(APPEND ${THREADING_LIST_VAR} "${CMAKE_MATCH_2}")
  endif()
endmacro()

macro(_GASNet_parse_conduit_makefile _GASNet_MAKEFILE _GASNet_THREADING)
  set(_TEMP_MAKEFILE ${CMAKE_CURRENT_BINARY_DIR}/FindGASNetParseConduitOpts.mak)
  if("${_GASNet_THREADING}" STREQUAL "parsync")
    file(WRITE ${_TEMP_MAKEFILE} "include ${_GASNet_MAKEFILE}")
  else()
    get_filename_component(MFDIR "${_GASNet_MAKEFILE}" DIRECTORY)
    file(WRITE ${_TEMP_MAKEFILE} "include ${_GASNet_MAKEFILE}
include ${MFDIR}/../gasnet_tools-${_GASNet_THREADING}.mak"
    )
  endif()
  file(APPEND ${_TEMP_MAKEFILE} "
gasnet-cc:
	@echo $(GASNET_CC)
gasnet-cflags:
	@echo $(GASNET_CPPFLAGS) $(GASNET_CFLAGS) $(GASNETTOOLS_CPPFLAGS) $(GASNETTOOLS_CFLAGS)
gasnet-cxx:
	@echo $(GASNET_CXX)
gasnet-cxxflags:
	@echo $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNETTOOLS_CPPFLAGS) $(GASNETTOOLS_CXXFLAGS)
gasnet-ld:
	@echo $(GASNET_LD)
gasnet-ldflags:
	@echo $(GASNET_LDFLAGS) $(GASNETTOOLS_LDFLAGS)
gasnet-libs:
	@echo $(GASNET_LIBS) $(GASNETTOOLS_LIBS)"
  )
  find_program(GASNet_MAKE_PROGRAM NAMES gmake make smake)
  mark_as_advanced(GASNet_MAKE_PROGRAM)
  if(NOT GASNet_MAKE_PROGRAM)
    message(WARNING "Unable to locate compatible make for parsing GASNet makefile options")
  else()
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-cflags
      OUTPUT_VARIABLE _GASNet_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-cxxflags
      OUTPUT_VARIABLE _GASNet_CXXFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-ld
      OUTPUT_VARIABLE _GASNet_LD
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    # Strip any arguments from _GASNet_LD
    string(REGEX REPLACE "^([^ ]+).*" "\\1" _GASNet_LD "${_GASNet_LD}")
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-ldflags
      OUTPUT_VARIABLE _GASNet_LDFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-libs
      OUTPUT_VARIABLE _GASNet_LIBS
      ERROR_VARIABLE _GASNet_LIBS_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE 
    )
    file(REMOVE ${_TEMP_MAKEFILE})
  endif()
endmacro()

macro(_GASNet_parse_flags INVAR FLAG OUTVAR)
  # Ignore the optimization flags
  string(REGEX REPLACE "-O[0-9]" "" INVAR2 "${${INVAR}}")
  string(REGEX MATCHALL "(^| +)${FLAG}([^ ]*)" OUTTMP "${INVAR2}")
  foreach(OPT IN LISTS OUTTMP)
    string(REGEX REPLACE "(^| +)${FLAG}([^ ]*)" "\\2" OPT "${OPT}")
    if(NOT (OPT STREQUAL "NDEBUG")) # NDEBUG should not get propogated
      list(FIND ${OUTVAR} "${OPT}" _I)
      if(_I EQUAL -1)
        list(APPEND ${OUTVAR} "${OPT}")
      endif()
    endif()
  endforeach()
endmacro()

function(_GASNet_create_component_target _GASNet_MAKEFILE COMPONENT_NAME)
  string(REGEX MATCH "^([^\\-]*)-([^\\-]*)$" COMPONENT_NAME "${COMPONENT_NAME}")
  _GASNet_parse_conduit_makefile(${_GASNet_MAKEFILE} ${CMAKE_MATCH_2})

  foreach(V _GASNet_CFLAGS _GASNet_CXXFLAGS _GASNet_LDFLAGS _GASNet_LIBS)
    _GASNet_parse_flags(${V} "-I" IDIRS)
    _GASNet_parse_flags(${V} "-D" DEFS)
    _GASNet_parse_flags(${V} "-L" LDIRS)
    _GASNet_parse_flags(${V} "-l" LIBS)
  endforeach()
  if(NOT LIBS)
    message(WARNING "Unable to find link libraries for gasnet-${COMPONENT_NAME}")
    return()
  endif()

  foreach(L IN LISTS LIBS)
    find_library(GASNet_${L}_LIBRARY ${L} PATHS ${LDIRS} NO_DEFAULT_PATH)
    if(NOT GASNet_${L}_LIBRARY)
      find_library(GASNet_${L}_LIBRARY ${L})
    endif()
    if(GASNet_${L}_LIBRARY)
      list(APPEND COMPONENT_DEPS "${GASNet_${L}_LIBRARY}")
    else()
      message(WARNING
        "Unable to locate GASNet ${COMPONENT_NAME} dependency ${L}"
      )
    endif()
    mark_as_advanced(GASNet_${L}_LIBRARY)
  endforeach()
  if(_GASNet_LD MATCHES "^(/.*/)?mpi[^/]*" AND NOT (_GASNet_LD STREQUAL CMAKE_C_COMPILER))
    set(MPI_C_COMPILER ${_GASNet_LD})
    find_package(MPI REQUIRED COMPONENTS C)
    list(APPEND COMPONENT_DEPS ${MPI_C_LIBRARIES})
  endif()
  add_library(GASNet::${COMPONENT_NAME} UNKNOWN IMPORTED)
  set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
    IMPORTED_LOCATION "${GASNet_gasnet-${COMPONENT_NAME}_LIBRARY}"
  )
  if(DEFS)
    set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "${DEFS}"
    )
  endif()
  if(IDIRS)
    set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${IDIRS}"
    )
  endif()
  if(COMPONENT_DEPS)
    set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
      INTERFACE_LINK_LIBRARIES "${COMPONENT_DEPS}"
    )
  endif()
endfunction()

# Needed for backwards compatibility with older cmake
macro(list_filter_include varname re)
  set(tmp_${varname})
  foreach(item IN LISTS ${varname})
    if(item MATCHES "${re}")
      list(APPEND tmp_${varname} ${item})
    endif()
  endforeach()
  set(${varname} ${tmp_${varname}})
  unset(tmp_${varname})
endmacro()
macro(list_filter_exclude varname re)
  set(tmp_${varname})
  foreach(item IN LISTS ${varname})
    if(NOT (item MATCHES "${re}"))
      list(APPEND tmp_${varname} ${item})
    endif()
  endforeach()
  set(${varname} ${tmp_${varname}})
  unset(tmp_${varname})
endmacro()

# This function takes an input list in the variable LVAR and a preferences list
# in PVAR, and sorts the list in LVAR by the items in PVAR
function(_sort_list_by_preference LVAR PVAR)
  string(REPLACE ";" "|" L_RE "${${LVAR}}")
  list_filter_include(${PVAR} "${L_RE}")
  string(REPLACE ";" "|" P_RE "${${PVAR}}")
  list_filter_exclude(${LVAR} "${P_RE}")
  set(L)
  list(APPEND L ${${PVAR}})
  list(APPEND L ${${LVAR}})
  set(${LVAR} ${L} PARENT_SCOPE)
endfunction()

if(NOT GASNet_FOUND AND NOT TARGET GASNet::GASNet)
  set(GASNet_ROOT_DIR "$ENV{GASNET_ROOT}" CACHE STRING "Root directory for GASNet")
  mark_as_advanced(GASNet_ROOT_DIR)
  if(GASNet_ROOT_DIR)
    set(_GASNet_FIND_INCLUDE_OPTS PATHS ${GASNet_ROOT_DIR}/include NO_DEFAULT_PATH)
  else()
    set(_GASNet_FIND_INCLUDE_OPTS HINTS ENV MPI_INCLUDE)
  endif()
  find_path(GASNet_INCLUDE_DIR gasnet.h ${_GASNet_FIND_INCLUDE_OPTS})

  # Make sure that all GASNet componets are found in the same install
  if(GASNet_INCLUDE_DIR)
    # Save the existing prefix options
    set(_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    set(_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})

    # Set new restrictive search paths
    get_filename_component(CMAKE_PREFIX_PATH "${GASNet_INCLUDE_DIR}" DIRECTORY)
    unset(CMAKE_LIBRARY_PATH)
    set(GASNet_ROOT_DIR ${CMAKE_PREFIX_PATH} CACHE STRING "Root directory for GASNet")

    # Limit the search to the discovered prefix path
    set(_GASNet_LIBRARY_FIND_OPTS
      NO_CMAKE_ENVIRONMENT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )

    # Look for the conduit specific headers
    set(GASNet_CONDUITS)
    set(GASNet_THREADING_OPTS)
    file(GLOB _GASNet_CONDUIT_MAKEFILES ${GASNet_INCLUDE_DIR}/*-conduit/*.mak)
    foreach(CMF IN LISTS _GASNet_CONDUIT_MAKEFILES)
      # Extract the component name from the makefile
      get_filename_component(_COMPONENT ${CMF} NAME_WE)

      # Seperate the filename components 
      _GASNet_parse_conduit_and_threading_names("${CMF}"
        GASNet_CONDUITS GASNet_THREADING_OPTS
      )

      # Create the component imported target
      _GASNet_create_component_target("${CMF}" ${_COMPONENT})

      if(NOT TARGET GASNet::${_COMPONENT})
        message(WARNING "Unable to create GASNet::${_COMPONENT} target")
      endif()
    endforeach()

    # Restore the existing prefix options
    set(CMAKE_PREFIX_PATH ${_CMAKE_PREFIX_PATH})
    set(CMAKE_LIBRARY_PATH ${_CMAKE_LIBRARY_PATH})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(GASNet
    FOUND_VAR GASNet_FOUND
    REQUIRED_VARS GASNet_INCLUDE_DIR GASNet_CONDUITS GASNet_THREADING_OPTS
  )
  if(GASNet_FOUND)
    # Reorder preferred conduits to place local preferences first
    set(GASNet_ALL_PREFERRED_CONDUITS
      "mxm;psm;gemini;aries;pami;ibv;shmem;portals4;ofi;mpi;udp;smp;ucx")
    set(GASNet_PREFERRED_CONDUITS "${GASNet_ALL_PREFERRED_CONDUITS}"
      CACHE STRING "")
    mark_as_advanced(GASNet_PREFERRED_CONDUITS)
    foreach(C IN LISTS GASNet_PREFERRED_CONDUITS)
      list(REMOVE_ITEM GASNet_ALL_PREFERRED_CONDUITS ${C})
    endforeach()
    list(APPEND GASNet_PREFERRED_CONDUITS ${GASNet_ALL_PREFERRED_CONDUITS})
    set(GASNet_PREFERRED_CONDUITS "${GASNet_PREFERRED_CONDUITS}"
      CACHE STRING "" FORCE)

    # Filter the conduit list to only contain what was found
    _sort_list_by_preference(GASNet_CONDUITS GASNet_PREFERRED_CONDUITS)

    # Stick them in the cache
    set(GASNet_CONDUITS ${GASNet_CONDUITS} CACHE INTERNAL "")
    set(GASNet_THREADING_OPTS ${GASNet_THREADING_OPTS} CACHE INTERNAL "")
    message(STATUS "Found GASNet Conduits: ${GASNet_CONDUITS}")
    message(STATUS "Found GASNet Threading models: ${GASNet_THREADING_OPTS}")
  endif()
endif()

# If found, use the CONDUIT and THREADING options to determine which target to
# use
if(GASNet_FOUND AND NOT TARGET GASNet::GASNet)
  if(NOT GASNet_CONDUIT)
    list(GET GASNet_CONDUITS 0 GASNet_CONDUIT)
  endif()
  set(GASNet_CONDUIT "${GASNet_CONDUIT}" CACHE STRING "GASNet communication conduit to use")
  mark_as_advanced(GASNet_CONDUIT)
  set_property(CACHE GASNet_CONDUIT PROPERTY STRINGS ${GASNet_CONDUITS})
  list(FIND GASNet_CONDUITS "${GASNet_CONDUIT}" _I)
  if(_I EQUAL -1)
    message(FATAL_ERROR "Invalid GASNet_CONDUIT setting.  Valid options are: ${GASNet_CONDUITS}")
  endif()

  set(GASNet_THREADING "${GASNet_THREADING}" CACHE STRING "GASNet Threading model to use")
  mark_as_advanced(GASNet_THREADING)
  set_property(CACHE GASNet_THREADING PROPERTY STRINGS ${GASNet_THREADING_OPTS})
  if(NOT GASNet_THREADING)
    list(GET GASNet_THREADING_OPTS 0 GASNet_THREADING)
    set(GASNet_THREADING "${GASNet_THREADING}")
  endif()
  list(FIND GASNet_THREADING_OPTS "${GASNet_THREADING}" _I)
  if(_I EQUAL -1)
    message(FATAL_ERROR "Invalid GASNet_THREADING setting.  Valid options are: ${GASNet_THREADINGS}")
  endif()

  if(NOT TARGET GASNet::${GASNet_CONDUIT}-${GASNet_THREADING})
    message(FATAL_ERROR "Unable to use selected CONDUIT-THREADING combination: ${GASNet_CONDUIT}-${GASNet_THREADING}")
  endif()
  message(STATUS "GASNet: Using ${GASNet_CONDUIT}-${GASNet_THREADING}")
  add_library(GASNet::GASNet INTERFACE IMPORTED)
  set_target_properties(GASNet::GASNet PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS GASNETI_BUG1389_WORKAROUND=1
    INTERFACE_LINK_LIBRARIES GASNet::${GASNet_CONDUIT}-${GASNet_THREADING}
  )
endif()