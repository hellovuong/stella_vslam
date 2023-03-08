file(GLOB common_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        )
file(GLOB common_H
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
        )
target_sources(${PROJECT_NAME}
        PRIVATE
        ${common_SRC}
        ${common_H}
        )

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${STELLA_VSLAM_INCLUDE_INSTALL_DIR}/feature/common_trt)