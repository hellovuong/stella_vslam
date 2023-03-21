FROM  ghcr.io/hellovuong/vslam_trt_base:cv-4.x-trt-8.4.3
LABEL maintainer="Long Vuong"

SHELL ["/bin/bash", "-c"]
ARG NUM_THREADS=2
COPY ${PWD} /stella_vslam/
WORKDIR /stella_vslam/
RUN set -x && \
  mkdir -p build && \
  cd build && \
  cmake \
    -DUSE_PANGOLIN_VIEWER=ON \
    -DBUILD_EXAMPLES=ON \
    -DDETERMINISTIC:BOOL=ON \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    .. && \
  make -j${NUM_THREADS} && \

ENTRYPOINT ["/bin/bash"]