# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_params.h
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor.h
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor_node.h
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_impl.h
               ${CMAKE_CURRENT_SOURCE_DIR}/extractor_factory.h
               ${CMAKE_CURRENT_SOURCE_DIR}/sp_extractor.h
               ${CMAKE_CURRENT_SOURCE_DIR}/base_extractor.h
               ${CMAKE_CURRENT_SOURCE_DIR}/sp_trt.h
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_params.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor_node.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/orb_impl.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/base_extractor.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/sp_extractor.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/sp_trt.cpp)

add_subdirectory(common_trt)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${STELLA_VSLAM_INCLUDE_INSTALL_DIR}/feature)
