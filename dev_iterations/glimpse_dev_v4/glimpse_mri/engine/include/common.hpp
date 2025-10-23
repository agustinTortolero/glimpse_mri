#pragma once
#include <vector>
#include <complex>
#include <string>
#include <iostream>
inline void dbg_head_impl(const char* tag){ std::cerr << "[DBG][" << tag << "] "; }
#define dbg_head(tag) dbg_head_impl(tag)
namespace pugi { class xml_node; }
int    xml_int(const pugi::xml_node& n, const char* path, int def=0);
double xml_double(const pugi::xml_node& n, const char* path, double def=0.0);
const char* xml_str(const pugi::xml_node& n, const char* path, const char* def="");
struct KsGrid { int coils=0, ny=0, nx=0; std::vector<std::complex<float>> host; };
bool dump_ismrmrd_metadata(const std::string& path);
