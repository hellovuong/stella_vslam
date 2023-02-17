# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/preintegrated_velocity_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/preintegrated_velocity_edge.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/bias_random_walk_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/bias_random_walk_edge.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/vertex_bias.h
               ${CMAKE_CURRENT_SOURCE_DIR}/vertex_bias.cpp)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${STELLA_VSLAM_INCLUDE_INSTALL_DIR}/optimize/internal/odometry)
