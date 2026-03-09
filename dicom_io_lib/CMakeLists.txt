cmake_minimum_required(VERSION 3.22)
project(dicom_io_lib LANGUAGES CXX)

message(STATUS "[DBG][DLL] Configuring dicom_io_lib with CMake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_VCPKG "Use vcpkg-provided DCMTK on Linux" OFF)
option(CODECS_ENABLED "Enable optional DCMTK codec libs" OFF)

add_library(dicom_io_lib SHARED
    src/dicom_io_lib.cpp
    include/dicom_api.h
)

target_include_directories(dicom_io_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(dicom_io_lib PRIVATE DICOM_IO_LIB_BUILD)

# predictable output dirs
set_target_properties(dicom_io_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# add "d" suffix for Debug on MSVC (dicom_io_libd.dll/.lib)
if(MSVC)
    set_target_properties(dicom_io_lib PROPERTIES DEBUG_POSTFIX "d")
endif()

# warnings / platform specifics
if(MSVC)
    target_compile_definitions(dicom_io_lib PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(dicom_io_lib PRIVATE /W4 /permissive- /Zc:__cplusplus)
elseif(UNIX)
    target_compile_options(dicom_io_lib PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(dicom_io_lib PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
endif()

function(add_optional_codec_libs target_name)
    if(CODECS_ENABLED)
        message(STATUS "[DBG][DLL] CODECS_ENABLED=ON -> linking optional codec libs")
        target_link_libraries(${target_name} PRIVATE
            dcmjpeg
            charls
            jpeg
            png
            tiff
            z
        )
    else()
        message(STATUS "[DBG][DLL] CODECS_ENABLED=OFF")
    endif()
endfunction()

# ======================= Dependency: DCMTK ===================================
# ----------------------- Windows (MSVC + vcpkg) ------------------------------
if(WIN32)
    if(NOT MSVC)
        message(FATAL_ERROR "[ERR][DLL] This project is configured for MSVC on Windows. Select an MSVC generator/toolchain.")
    endif()

    message(STATUS "[DBG][DLL] Platform = Windows (MSVC + vcpkg)")

    set(VCPKG_ROOT_CANDIDATES "")
    if(DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
        list(APPEND VCPKG_ROOT_CANDIDATES "$ENV{VCPKG_ROOT}")
    endif()

    if(DEFINED ENV{LOCALAPPDATA} AND NOT "$ENV{LOCALAPPDATA}" STREQUAL "")
        list(APPEND VCPKG_ROOT_CANDIDATES "$ENV{LOCALAPPDATA}/vcpkg")
    endif()

    if(DEFINED ENV{USERPROFILE} AND NOT "$ENV{USERPROFILE}" STREQUAL "")
        list(APPEND VCPKG_ROOT_CANDIDATES "$ENV{USERPROFILE}/vcpkg")
    endif()

    list(APPEND VCPKG_ROOT_CANDIDATES
        "C:/src/vcpkg"
        "C:/vcpkg"
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg"
    )

    if(NOT DEFINED VCPKG_TRIPLET OR VCPKG_TRIPLET STREQUAL "")
        set(VCPKG_TRIPLET "x64-windows")
    endif()

    message(STATUS "[DBG][DLL] Triplet = ${VCPKG_TRIPLET}")
    message(STATUS "[DBG][DLL] Roots to probe = ${VCPKG_ROOT_CANDIDATES}")

    set(MATCHED_VCPKG_ROOT "")
    foreach(candidate IN LISTS VCPKG_ROOT_CANDIDATES)
        set(_hdr "${candidate}/installed/${VCPKG_TRIPLET}/include/dcmtk/config/osconfig.h")
        message(STATUS "[DBG][DLL] check: ${_hdr}")
        if(EXISTS "${_hdr}")
            set(MATCHED_VCPKG_ROOT "${candidate}")
        endif()
    endforeach()

    if(MATCHED_VCPKG_ROOT STREQUAL "")
        message(FATAL_ERROR
            "[ERR][DLL] dcmtk/config/osconfig.h not found for triplet: ${VCPKG_TRIPLET}\n"
            "[TIP][DLL] Install DCMTK: vcpkg install dcmtk:${VCPKG_TRIPLET}\n"
            "[TIP][DLL] Or set VCPKG_ROOT in the environment."
        )
    endif()

    set(VCPKG_INSTALLED "${MATCHED_VCPKG_ROOT}/installed/${VCPKG_TRIPLET}")
    set(DCMTK_INC "${VCPKG_INSTALLED}/include")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(DCMTK_LIB "${VCPKG_INSTALLED}/debug/lib")
    else()
        set(DCMTK_LIB "${VCPKG_INSTALLED}/lib")
    endif()

    message(STATUS "[DBG][DLL] Using VCPKG_ROOT = ${MATCHED_VCPKG_ROOT}")
    message(STATUS "[DBG][DLL] DCMTK_INC       = ${DCMTK_INC}")
    message(STATUS "[DBG][DLL] DCMTK_LIB       = ${DCMTK_LIB}")

    target_include_directories(dicom_io_lib PRIVATE "${DCMTK_INC}")
    target_link_directories(dicom_io_lib PRIVATE "${DCMTK_LIB}")

    target_link_libraries(dicom_io_lib PRIVATE
        dcmimgle
        dcmimage
        dcmdata
        ofstd
        oflog
        ws2_32
        wsock32
        iphlpapi
        advapi32
        user32
    )

    add_optional_codec_libs(dicom_io_lib)

# ------------------------- Linux / Unix --------------------------------------
elseif(UNIX)
    set(USE_VCPKG "${USE_VCPKG}" CACHE BOOL "Use vcpkg-provided DCMTK on Linux")

    if(USE_VCPKG)
        message(STATUS "[DBG][DLL] Platform = Linux (vcpkg)")

        if(NOT DEFINED VCPKG_ROOT OR VCPKG_ROOT STREQUAL "")
            if(DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
                set(VCPKG_ROOT "$ENV{VCPKG_ROOT}")
            else()
                set(VCPKG_ROOT "/usr/local/vcpkg")
            endif()
        endif()

        if(NOT DEFINED VCPKG_TRIPLET OR VCPKG_TRIPLET STREQUAL "")
            execute_process(
                COMMAND uname -m
                OUTPUT_VARIABLE _uname_m
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(_uname_m STREQUAL "aarch64")
                set(VCPKG_TRIPLET "arm64-linux")
            elseif(_uname_m STREQUAL "armv7l")
                set(VCPKG_TRIPLET "arm-linux")
            elseif(_uname_m STREQUAL "x86_64")
                set(VCPKG_TRIPLET "x64-linux")
            else()
                message(FATAL_ERROR "[ERR][DLL] Unsupported uname -m for default triplet mapping: ${_uname_m}")
            endif()
        endif()

        set(VCPKG_INSTALLED "${VCPKG_ROOT}/installed/${VCPKG_TRIPLET}")
        set(DCMTK_INC "${VCPKG_INSTALLED}/include")

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(DCMTK_LIB "${VCPKG_INSTALLED}/debug/lib")
        else()
            set(DCMTK_LIB "${VCPKG_INSTALLED}/lib")
        endif()

        if(NOT EXISTS "${DCMTK_INC}/dcmtk/config/osconfig.h")
            message(FATAL_ERROR
                "[ERR][DLL] DCMTK header NOT found: ${DCMTK_INC}/dcmtk/config/osconfig.h\n"
                "[TIP][DLL] Fix: vcpkg install dcmtk:${VCPKG_TRIPLET} or use system pkg via pkg-config"
            )
        endif()

        message(STATUS "[DBG][DLL] VCPKG_ROOT = ${VCPKG_ROOT}")
        message(STATUS "[DBG][DLL] TRIPLET    = ${VCPKG_TRIPLET}")
        message(STATUS "[DBG][DLL] INC/LIB    = ${DCMTK_INC} | ${DCMTK_LIB}")

        target_include_directories(dicom_io_lib PRIVATE "${DCMTK_INC}")
        target_link_directories(dicom_io_lib PRIVATE "${DCMTK_LIB}")

        target_link_libraries(dicom_io_lib PRIVATE
            dcmimgle
            dcmimage
            dcmdata
            ofstd
            oflog
            pthread
        )

        add_optional_codec_libs(dicom_io_lib)

    else()
        message(STATUS "[DBG][DLL] Platform = Linux (pkg-config: dcmtk preferred)")

        find_package(PkgConfig REQUIRED)
        pkg_check_modules(DCMTK QUIET IMPORTED_TARGET dcmtk)

        if(TARGET PkgConfig::DCMTK)
            message(STATUS "[DBG][DLL] Using pkg-config: dcmtk")
            target_link_libraries(dicom_io_lib PRIVATE PkgConfig::DCMTK)
        else()
            message(WARNING "[WRN][DLL] dcmtk.pc not found. Falling back to manual include+libs.")
            target_include_directories(dicom_io_lib PRIVATE /usr/include/dcmtk)
            target_link_libraries(dicom_io_lib PRIVATE
                dcmimgle
                dcmimage
                dcmdata
                ofstd
                oflog
                pthread
                z
            )
            add_optional_codec_libs(dicom_io_lib)
        endif()
    endif()
endif()

# -------- final debug prints --------------------------------------------------
message(STATUS "[DBG][DLL] CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "[DBG][DLL] CMAKE_BINARY_DIR   = ${CMAKE_BINARY_DIR}")
message(STATUS "[DBG][DLL] BUILD_TYPE         = ${CMAKE_BUILD_TYPE}")
message(STATUS "[DBG][DLL] USE_VCPKG          = ${USE_VCPKG}")
message(STATUS "[DBG][DLL] CODECS_ENABLED     = ${CODECS_ENABLED}")