// engine/include/common.hpp
#pragma once

// ---- Windows macro hygiene (safe on Linux) ---------------------------------
#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#  define NOMINMAX 1              // prevent min/max macros
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN 1   // shrink Windows.h if it appears
#endif
#endif

#include <vector>
#include <complex>
#include <string>
#include <iostream>                 // for dbg_head printing

// In case any TU still saw min/max as macros before this header
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

namespace pugi { class xml_node; }


extern void engine_log_line(const char* line);

// --- tiny debug prefix helper used across the engine -------------------------
inline void dbg_head_impl(const char* tag) { std::cerr << "[DBG][" << tag << "] "; }
#define dbg_head(tag) dbg_head_impl(tag)

// Lightweight k-space holder
struct KsGrid {
    int coils = 0, ny = 0, nx = 0;
    std::vector<std::complex<float>> host;
};

// XML helpers (implemented in common.cpp)
int         xml_int(const pugi::xml_node& n, const char* path, int def = 0);
double      xml_double(const pugi::xml_node& n, const char* path, double def = 0.0);
const char* xml_str(const pugi::xml_node& n, const char* path, const char* def = "");

// Optional ISMRMRD metadata dump (implemented in common.cpp)
bool dump_ismrmrd_metadata(const std::string& path);

// Engine-wide logging bridge (implemented in common.cpp)
void engine_log_line(const char* line);
void engine_logf(const char* fmt, ...);
