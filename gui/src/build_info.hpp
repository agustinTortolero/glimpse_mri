#pragma once

// ------------------------------------------------------------
// build_info.hpp
// Safe fallback macros for build/version fingerprinting.
// Values are normally injected via qmake .pro using DEFINES += ...
// ------------------------------------------------------------

// GUI / build
#ifndef GUI_VERSION_STR
#define GUI_VERSION_STR "Unknown"
#endif

#ifndef BUILD_TYPE_STR
#define BUILD_TYPE_STR "Unknown"
#endif

#ifndef BUILD_TIMESTAMP_STR
#define BUILD_TIMESTAMP_STR "Unknown"
#endif

// Git
#ifndef GIT_SHA_STR
#define GIT_SHA_STR "Unknown"
#endif

#ifndef GIT_DESCRIBE_STR
#define GIT_DESCRIBE_STR "Unknown"
#endif

// Qt / qmake (from .pro)
#ifndef QT_BUILD_VERSION_STR
#define QT_BUILD_VERSION_STR "Unknown"
#endif

#ifndef QMAKE_BUILD_VERSION_STR
#define QMAKE_BUILD_VERSION_STR "Unknown"
#endif

#ifndef QMAKE_PATH_STR
#define QMAKE_PATH_STR "Unknown"
#endif

// Optional: if later you decide to add more metadata, keep these ready
#ifndef COMPILER_ID_STR
#define COMPILER_ID_STR "Unknown"
#endif

#ifndef COMPILER_VERSION_STR
#define COMPILER_VERSION_STR "Unknown"
#endif
