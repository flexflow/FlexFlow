add_subdirectory(pcg)
# add_subdirectory(ffc)
#add_subdirectory(ffr)
add_subdirectory(op-meta)
#add_subdirectory(op-impl)
add_subdirectory(utils)

set(FLEXFLOW_LIBS ${FLEXFLOW_LIBS} PARENT_SCOPE)
set(FLEXFLOW_INCLUDE_DIRS ${FLEXFLOW_INCLUDE_DIRS} PARENT_SCOPE)
