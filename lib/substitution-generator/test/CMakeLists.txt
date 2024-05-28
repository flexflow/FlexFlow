add_custom_target(copy-test-makefile ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/graph_subst_3_v2.json)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/graph_subst_3_v2.json
                   COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/substitutions/graph_subst_3_v2.json
                                                    ${CMAKE_CURRENT_BINARY_DIR}/graph_subst_3_v2.json
                   DEPENDS ${PROJECT_SOURCE_DIR}/substitutions/graph_subst_3_v2.json)

ff_add_test_executable(
  NAME
    substitution-generator-tests
  SRC_PATTERNS
    substitution-generator/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    doctest
    substitution-generator
    utils-test-common
)
add_dependencies(substitution-generator-tests copy-test-makefile)
