cmake_minimum_required(VERSION 3.25)
project(top.matrix_product LANGUAGES CXX)

# Configuration spécifique pour macOS ARM
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

# Configuration de Kokkos avec OpenMP
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend" FORCE)

include(FetchContent)
FetchContent_Declare(Kokkos GIT_REPOSITORY https://github.com/kokkos/kokkos GIT_TAG 4.6.00)
FetchContent_MakeAvailable(Kokkos)
FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt GIT_TAG 11.1.4)
FetchContent_MakeAvailable(fmt)

add_subdirectory(src)
