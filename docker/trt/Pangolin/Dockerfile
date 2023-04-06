FROM  ghcr.io/hellovuong/vslam_trt_base:cv-4.x-trt-8.4.3
LABEL maintainer="Long Vuong"

SHELL ["/bin/bash", "-c"]
ARG NUM_THREADS=2
COPY ${PWD} /stella_vslam/
WORKDIR /stella_vslam/
RUN set -x && \
  mkdir -p build && \
  cmake -B build \
    -GNinja \
    -DBUILD_EXAMPLES=ON \
    -DUSE_GTSAM=ON \
    -DUSE_OPENMP=ON \
    -DDETERMINISTIC=ON \
    -DENABLE_TRACE_LEVEL_LOG=ON \
    -DUSE_OPENMP=ON \
    -DUSE_STACK_TRACE_LOGGER:BOOL=ON \
    -DUSE_PANGOLIN_VIEWER:BOOL=ON \
    -DBUILD_TESTS:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release && \
  cmake --build -j${NUM_THREADS}

ENTRYPOINT ["/bin/bash"]