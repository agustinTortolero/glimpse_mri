cmake_minimum_required(VERSION 3.22)
project(mri_engine LANGUAGES C CXX)

message(STATUS "[eng] Configuring mri_engine with CMake")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CPU-first by default.
# AUTO = enable CUDA only if a usable toolkit/compiler is found at build time
# ON   = require CUDA and fail if it is unavailable
# OFF  = force CPU-only build
set(GLIMPSE_CUDA_MODE "AUTO" CACHE STRING "CUDA mode: AUTO, ON, OFF")
set_property(CACHE GLIMPSE_CUDA_MODE PROPERTY STRINGS AUTO ON OFF)

option(ENGINE_CODECS_ENABLED "Link optional DCMTK codec libs when available" OFF)

message(STATUS "[eng] GLIMPSE_CUDA_MODE=${GLIMPSE_CUDA_MODE}")
message(STATUS "[eng] CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "[eng] CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Compatibility with older script flags:
#   -DENABLE_CUDA=ON/OFF
#   -DUSE_CUDA=ON/OFF
#   -DGLIMPSE_ENABLE_CUDA=ON/OFF
# If GLIMPSE_CUDA_MODE is still AUTO, map those legacy flags into AUTO/ON/OFF behavior.
set(_LEGACY_CUDA_REQUESTED OFF)
set(_LEGACY_CUDA_DEFINED OFF)

if(DEFINED ENABLE_CUDA)
    set(_LEGACY_CUDA_DEFINED ON)
    if(ENABLE_CUDA)
        set(_LEGACY_CUDA_REQUESTED ON)
    endif()
endif()

if(DEFINED USE_CUDA)
    set(_LEGACY_CUDA_DEFINED ON)
    if(USE_CUDA)
        set(_LEGACY_CUDA_REQUESTED ON)
    endif()
endif()

if(DEFINED GLIMPSE_ENABLE_CUDA)
    set(_LEGACY_CUDA_DEFINED ON)
    if(GLIMPSE_ENABLE_CUDA)
        set(_LEGACY_CUDA_REQUESTED ON)
    endif()
endif()

if(GLIMPSE_CUDA_MODE STREQUAL "AUTO" AND _LEGACY_CUDA_DEFINED)
    if(_LEGACY_CUDA_REQUESTED)
        set(GLIMPSE_CUDA_MODE "ON")
        message(STATUS "[eng][cuda] legacy flags requested CUDA -> forcing GLIMPSE_CUDA_MODE=ON")
    else()
        set(GLIMPSE_CUDA_MODE "OFF")
        message(STATUS "[eng][cuda] legacy flags disabled CUDA -> forcing GLIMPSE_CUDA_MODE=OFF")
    endif()
endif()

# ---- recursive source collection (matches qmake files(..., true)) ----
file(GLOB_RECURSE ENGINE_C_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

file(GLOB_RECURSE ENGINE_CPP_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

file(GLOB_RECURSE ENGINE_HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

file(GLOB_RECURSE ENGINE_CUDA_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

list(LENGTH ENGINE_C_SOURCES ENGINE_C_COUNT)
list(LENGTH ENGINE_CPP_SOURCES ENGINE_CPP_COUNT)
list(LENGTH ENGINE_HEADERS ENGINE_H_COUNT)
list(LENGTH ENGINE_CUDA_SOURCES ENGINE_CUDA_COUNT)

message(STATUS "[eng] C sources   = ${ENGINE_C_COUNT}")
message(STATUS "[eng] C++ sources = ${ENGINE_CPP_COUNT}")
message(STATUS "[eng] headers     = ${ENGINE_H_COUNT}")
message(STATUS "[eng] CUDA files  = ${ENGINE_CUDA_COUNT}")

add_library(mri_engine SHARED
    ${ENGINE_C_SOURCES}
    ${ENGINE_CPP_SOURCES}
    ${ENGINE_HEADERS}
)

# Keep Linux build outputs in the engine build dir.
# This matches the new scripts/core flow.
set_target_properties(mri_engine PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(mri_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# warnings
if(MSVC)
    target_compile_options(mri_engine PRIVATE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(mri_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- common linux libs ----
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(mri_engine PRIVATE Threads::Threads dl z)
endif()

# ---- optional OpenMP ----
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "[eng] OpenMP found -> enabling OpenMP::OpenMP_CXX")
    target_link_libraries(mri_engine PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "[eng] OpenMP not found -> continuing without OpenMP")
endif()

# ---- pkg-config deps that exist on Jetson/Linux ----
find_package(PkgConfig REQUIRED)

pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(PUGIXML REQUIRED IMPORTED_TARGET pugixml)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(HDF5_PC QUIET IMPORTED_TARGET hdf5)

target_link_libraries(mri_engine PRIVATE
    PkgConfig::OPENCV
    PkgConfig::PUGIXML
    PkgConfig::FFTW3F
)

if(TARGET PkgConfig::HDF5_PC)
    message(STATUS "[eng][hdf5] base hdf5 resolved via pkg-config")
    target_link_libraries(mri_engine PRIVATE PkgConfig::HDF5_PC)
else()
    message(WARNING "[eng][hdf5] hdf5.pc not found; relying on manual HDF5 HL/C++ libs below")
endif()

# ---- HDF5 HL/C++: pkg-config often misses these; link manually ----
set(HDF5_INCDIR "/usr/include/hdf5/serial" CACHE PATH "HDF5 serial include dir")

set(_HDF5_LIBDIR_DEFAULT "")
if(DEFINED CMAKE_LIBRARY_ARCHITECTURE AND EXISTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/hdf5/serial")
    set(_HDF5_LIBDIR_DEFAULT "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/hdf5/serial")
elseif(EXISTS "/usr/lib/aarch64-linux-gnu/hdf5/serial")
    set(_HDF5_LIBDIR_DEFAULT "/usr/lib/aarch64-linux-gnu/hdf5/serial")
elseif(EXISTS "/usr/lib/x86_64-linux-gnu/hdf5/serial")
    set(_HDF5_LIBDIR_DEFAULT "/usr/lib/x86_64-linux-gnu/hdf5/serial")
else()
    set(_HDF5_LIBDIR_DEFAULT "/usr/lib/hdf5/serial")
endif()

set(HDF5_LIBDIR "${_HDF5_LIBDIR_DEFAULT}" CACHE PATH "HDF5 serial lib dir")

message(STATUS "[eng][hdf5] manual hl/cpp: inc=${HDF5_INCDIR} lib=${HDF5_LIBDIR}")

target_include_directories(mri_engine PRIVATE "${HDF5_INCDIR}")
target_link_directories(mri_engine PRIVATE "${HDF5_LIBDIR}")
target_link_libraries(mri_engine PRIVATE
    hdf5_hl
    hdf5_cpp
    hdf5_hl_cpp
)

# ---- ISMRMRD: manual link ----
set(ISMRMRD_INCDIR "/usr/include" CACHE PATH "ISMRMRD include dir (parent of the ismrmrd/ folder)")

set(_ISMRMRD_LIBDIR_DEFAULT "")
if(DEFINED CMAKE_LIBRARY_ARCHITECTURE AND EXISTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    set(_ISMRMRD_LIBDIR_DEFAULT "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
elseif(EXISTS "/usr/lib/aarch64-linux-gnu")
    set(_ISMRMRD_LIBDIR_DEFAULT "/usr/lib/aarch64-linux-gnu")
elseif(EXISTS "/usr/lib/x86_64-linux-gnu")
    set(_ISMRMRD_LIBDIR_DEFAULT "/usr/lib/x86_64-linux-gnu")
else()
    set(_ISMRMRD_LIBDIR_DEFAULT "/usr/lib")
endif()

set(ISMRMRD_LIBDIR "${_ISMRMRD_LIBDIR_DEFAULT}" CACHE PATH "ISMRMRD lib dir")

message(STATUS "[eng][ismrmrd] manual link: inc=${ISMRMRD_INCDIR} lib=${ISMRMRD_LIBDIR}")

target_include_directories(mri_engine PRIVATE "${ISMRMRD_INCDIR}")
target_link_directories(mri_engine PRIVATE "${ISMRMRD_LIBDIR}")
target_link_libraries(mri_engine PRIVATE ismrmrd)

# ---- optional DCMTK codec libs ----
function(add_optional_dcmtk_codec_libs target_name)
    if(ENGINE_CODECS_ENABLED)
        message(STATUS "[eng][dcmtk] ENGINE_CODECS_ENABLED=ON -> linking codec libs")
        target_link_libraries(${target_name} PRIVATE
            dcmjpeg
            charls
            jpeg
            png
            tiff
            z
        )
    else()
        message(STATUS "[eng][dcmtk] ENGINE_CODECS_ENABLED=OFF")
    endif()
endfunction()

add_optional_dcmtk_codec_libs(mri_engine)

# ---- CUDA detection / enablement ----
set(_GLIMPSE_CUDA_ENABLED OFF)
set(_GLIMPSE_CUDA_REASON "disabled by configuration")

string(TOUPPER "${GLIMPSE_CUDA_MODE}" _GLIMPSE_CUDA_MODE_UPPER)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _GLIMPSE_SYSTEM_PROCESSOR_LOWER)
set(_GLIMPSE_IS_AARCH64 OFF)
if(_GLIMPSE_SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
    set(_GLIMPSE_IS_AARCH64 ON)
endif()

if(NOT _GLIMPSE_CUDA_MODE_UPPER STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "[eng][cuda] CUDAToolkit found: ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
        if(NOT CMAKE_CUDA_COMPILER AND DEFINED CUDAToolkit_NVCC_EXECUTABLE AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
            set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler")
            message(STATUS "[eng][cuda] probing CUDA with nvcc from CUDAToolkit: ${CMAKE_CUDA_COMPILER}")
        endif()
    else()
        message(STATUS "[eng][cuda] CUDAToolkit package not found during probe")
    endif()

    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        message(STATUS "[eng][cuda] CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")
        enable_language(CUDA)

        if(CUDAToolkit_FOUND)
            set(_GLIMPSE_CUDA_ENABLED ON)
            set(_GLIMPSE_CUDA_REASON "CUDAToolkit found")
        else()
            set(_GLIMPSE_CUDA_REASON "CUDA compiler found but CUDAToolkit package not resolved")
        endif()
    else()
        set(_GLIMPSE_CUDA_REASON "CUDA compiler not found")
    endif()
endif()

if(_GLIMPSE_CUDA_MODE_UPPER STREQUAL "ON" AND NOT _GLIMPSE_CUDA_ENABLED)
    message(FATAL_ERROR "[eng][cuda] GLIMPSE_CUDA_MODE=ON but CUDA is unavailable: ${_GLIMPSE_CUDA_REASON}")
endif()

if(_GLIMPSE_CUDA_ENABLED AND ENGINE_CUDA_COUNT GREATER 0)
    message(STATUS "[eng][cuda] enabled: ${_GLIMPSE_CUDA_REASON}")

    target_sources(mri_engine PRIVATE ${ENGINE_CUDA_SOURCES})
    target_compile_definitions(mri_engine PRIVATE ENGINE_HAS_CUDA=1)
    set_target_properties(mri_engine PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        if(_GLIMPSE_IS_AARCH64)
            set_property(TARGET mri_engine PROPERTY CUDA_ARCHITECTURES 87)
            message(STATUS "[eng][cuda] defaulting CUDA_ARCHITECTURES=87 for Jetson/aarch64")
        else()
            set_property(TARGET mri_engine PROPERTY CUDA_ARCHITECTURES 75)
            message(WARNING "[eng][cuda] CMAKE_CUDA_ARCHITECTURES not set; defaulting to 75. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> for desktop GPUs.")
        endif()
    else()
        set_property(TARGET mri_engine PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        message(STATUS "[eng][cuda] using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    target_compile_options(mri_engine PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )

    target_link_libraries(mri_engine PRIVATE
        CUDA::cudart
        CUDA::cufft
        CUDA::cublas
    )
else()
    if(ENGINE_CUDA_COUNT GREATER 0)
        message(STATUS "[eng][cuda] disabled: ${_GLIMPSE_CUDA_REASON}")
    else()
        message(STATUS "[eng][cuda] no .cu sources found -> CPU-only build")
    endif()
endif()

message(STATUS "[eng] LIBRARY_OUTPUT_DIRECTORY=$<TARGET_FILE_DIR:mri_engine>")
message(STATUS "[eng] pkg-config deps: opencv4 pugixml fftw3f hdf5(optional)")
