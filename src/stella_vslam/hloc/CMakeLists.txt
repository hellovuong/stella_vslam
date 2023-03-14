# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hf_net.h
        ${CMAKE_CURRENT_SOURCE_DIR}/hf_net.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hloc_database.h
        ${CMAKE_CURRENT_SOURCE_DIR}/hloc_database.cpp)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${STELLA_VSLAM_INCLUDE_INSTALL_DIR}/hloc)