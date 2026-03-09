cmake_minimum_required(VERSION 3.22)
project(glimpseMRI LANGUAGES CXX)

message(STATUS "[gui] Configuring glimpseMRI with CMake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# ------------------------------------------------------------
# Qt / OpenCV
# ------------------------------------------------------------
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

message(STATUS "[gui] Qt6 version      = ${Qt6_VERSION}")
message(STATUS "[gui] OpenCV version   = ${OpenCV_VERSION}")
message(STATUS "[gui] Build type       = ${CMAKE_BUILD_TYPE}")
message(STATUS "[gui] System processor = ${CMAKE_SYSTEM_PROCESSOR}")

# ------------------------------------------------------------
# Build fingerprint (replacement for qmake-injected macros)
# ------------------------------------------------------------
set(GUI_VERSION "1.1.0")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

execute_process(
    COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse --short=12 HEAD
    OUTPUT_VARIABLE GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_SHA)
    set(GIT_SHA "Unknown")
endif()

execute_process(
    COMMAND git -C ${CMAKE_SOURCE_DIR} describe --tags --always --dirty --abbrev=12
    OUTPUT_VARIABLE GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_DESCRIBE)
    set(GIT_DESCRIBE "Unknown")
endif()

string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%S")

set(QT_BUILD_VERSION "${Qt6_VERSION}")
set(QMAKE_BUILD_VERSION "cmake")
set(QMAKE_PATH "cmake")

message(STATUS "[gui][ver] GUI_VERSION         = ${GUI_VERSION}")
message(STATUS "[gui][ver] BUILD_TYPE          = ${CMAKE_BUILD_TYPE}")
message(STATUS "[gui][ver] GIT_SHA             = ${GIT_SHA}")
message(STATUS "[gui][ver] GIT_DESCRIBE        = ${GIT_DESCRIBE}")
message(STATUS "[gui][ver] BUILD_TIMESTAMP     = ${BUILD_TIMESTAMP}")
message(STATUS "[gui][ver] QT_BUILD_VERSION    = ${QT_BUILD_VERSION}")
message(STATUS "[gui][ver] QMAKE_BUILD_VERSION = ${QMAKE_BUILD_VERSION}")
message(STATUS "[gui][ver] QMAKE_PATH          = ${QMAKE_PATH}")

# ------------------------------------------------------------
# Sources / headers / resources
# ------------------------------------------------------------
set(GUI_SOURCES
    model/io.cpp
    src/main.cpp
    controller/app_controller.cpp
    view/mainwindow.cpp
    view/progress_splash.cpp
    assets/assets.qrc
)

set(GUI_HEADERS
    model/engine_api.h
    model/io.hpp
    model/dicom_interface.hpp
    controller/app_controller.hpp
    src/build_info.hpp
    src/logger.hpp
    src/image_utils.hpp
    view/mainwindow.hpp
    view/progress_splash.hpp
)

add_executable(glimpseMRI
    ${GUI_SOURCES}
    ${GUI_HEADERS}
)

target_include_directories(glimpseMRI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/controller
    ${CMAKE_CURRENT_SOURCE_DIR}/view
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# Linux-side feature defines carried from qmake logic
# ------------------------------------------------------------
if(UNIX AND NOT WIN32)
    target_compile_definitions(glimpseMRI PRIVATE
        HAVE_ISMRMRD
        IO_ENABLE_DCMJPEG=1
        IO_ENABLE_DCMJPLS=1
        IO_ENABLE_DCMJ2K=0
    )

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_definitions(glimpseMRI PRIVATE GLIMPSE_JETSON)
        message(STATUS "[gui] Defining GLIMPSE_JETSON for aarch64/arm64 build")
    endif()
endif()

# ------------------------------------------------------------
# Build info macros expected by src/build_info.hpp
# ------------------------------------------------------------
target_compile_definitions(glimpseMRI PRIVATE
    GUI_VERSION_STR="${GUI_VERSION}"
    BUILD_TYPE_STR="${CMAKE_BUILD_TYPE}"
    GIT_SHA_STR="${GIT_SHA}"
    GIT_DESCRIBE_STR="${GIT_DESCRIBE}"
    BUILD_TIMESTAMP_STR="${BUILD_TIMESTAMP}"
    QT_BUILD_VERSION_STR="${QT_BUILD_VERSION}"
    QMAKE_BUILD_VERSION_STR="${QMAKE_BUILD_VERSION}"
    QMAKE_PATH_STR="${QMAKE_PATH}"
    COMPILER_ID_STR="${CMAKE_CXX_COMPILER_ID}"
    COMPILER_VERSION_STR="${CMAKE_CXX_COMPILER_VERSION}"
)

# ------------------------------------------------------------
# Locate engine + dicom libs from earlier build steps
# These are passed by scripts/core/build_modules.sh
# ------------------------------------------------------------
set(GLIMPSE_ENGINE_BUILD_DIR "" CACHE PATH "Path to build_engine_<TYPE>")
set(GLIMPSE_DICOM_BUILD_DIR  "" CACHE PATH "Path to build_dicom_<TYPE>")

message(STATUS "[gui] GLIMPSE_ENGINE_BUILD_DIR = ${GLIMPSE_ENGINE_BUILD_DIR}")
message(STATUS "[gui] GLIMPSE_DICOM_BUILD_DIR  = ${GLIMPSE_DICOM_BUILD_DIR}")

find_library(GLIMPSE_ENGINE_LIB
    NAMES mri_engine
    PATHS
        ${GLIMPSE_ENGINE_BUILD_DIR}
        ${GLIMPSE_ENGINE_BUILD_DIR}/lib
    NO_DEFAULT_PATH
)

find_library(GLIMPSE_DICOM_LIB
    NAMES dicom_io_lib
    PATHS
        ${GLIMPSE_DICOM_BUILD_DIR}
        ${GLIMPSE_DICOM_BUILD_DIR}/lib
    NO_DEFAULT_PATH
)

message(STATUS "[gui] GLIMPSE_ENGINE_LIB = ${GLIMPSE_ENGINE_LIB}")
message(STATUS "[gui] GLIMPSE_DICOM_LIB  = ${GLIMPSE_DICOM_LIB}")

if(NOT GLIMPSE_ENGINE_LIB)
    message(FATAL_ERROR "[gui][ERR] Could not find mri_engine. Pass -DGLIMPSE_ENGINE_BUILD_DIR=...")
endif()

if(NOT GLIMPSE_DICOM_LIB)
    message(FATAL_ERROR "[gui][ERR] Could not find dicom_io_lib. Pass -DGLIMPSE_DICOM_BUILD_DIR=...")
endif()

# ------------------------------------------------------------
# Link
# ------------------------------------------------------------
target_link_libraries(glimpseMRI PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${HDF5_LIBRARIES}
    ${GLIMPSE_ENGINE_LIB}
    ${GLIMPSE_DICOM_LIB}
)

# ------------------------------------------------------------
# RPATH: let the executable find ../lib in packaged bundles
# and also sibling libs during local Linux runs
# ------------------------------------------------------------
set_target_properties(glimpseMRI PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
)

message(STATUS "[gui] CMake configuration for glimpseMRI ready")