// src/common.cpp — portable helpers (Windows / Ubuntu / Raspberry Pi)
// Uses dbg_head(...) from common.hpp (do not redefine here)

#include "common.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

// --- Feature flags (override in your build configuration if needed) ----------
#if !defined(ENGINE_HAS_ISMRMRD)
#define ENGINE_HAS_ISMRMRD 1
#endif

// PugiXML is header-only and required here for XML helpers
#include <pugixml.hpp>

// ISMRMRD only if enabled
#if ENGINE_HAS_ISMRMRD
#include <ismrmrd/dataset.h>
#endif

// -----------------------------------------------------------------------------
// Small utility: walk a slash-separated path like "a/b/c" starting from node n.
// Returns an empty node if not found.
// -----------------------------------------------------------------------------
namespace {
    static pugi::xml_node walk_path(const pugi::xml_node& n, const char* path) {
        if (!path || !*path) return n;
        const char* p = path;
        pugi::xml_node cur = n;

        while (*p) {
            // read token until '/' or end
            const char* start = p;
            while (*p && *p != '/') ++p;
            std::string token(start, static_cast<size_t>(p - start));
            if (!token.empty()) {
                cur = cur.child(token.c_str());
                if (!cur) return pugi::xml_node(); // not found
            }
            if (*p == '/') ++p; // skip '/'
        }
        return cur;
    }
} // namespace

// -----------------------------------------------------------------------------
// XML getters (declared in common.hpp)
// -----------------------------------------------------------------------------
int xml_int(const pugi::xml_node& n, const char* path, int def) {
    auto t = walk_path(n, path);
    if (!t) {
        dbg_head("XML"); std::cerr << "xml_int('" << (path ? path : "") << "') -> <missing>, default=" << def << "\n";
        return def;
    }
    auto txt = t.text();
    if (!txt) {
        dbg_head("XML"); std::cerr << "xml_int('" << (path ? path : "") << "') -> <no-text>, default=" << def << "\n";
        return def;
    }
    int v = txt.as_int(def);
    dbg_head("XML"); std::cerr << "xml_int('" << (path ? path : "") << "') = " << v << "\n";
    return v;
}

double xml_double(const pugi::xml_node& n, const char* path, double def) {
    auto t = walk_path(n, path);
    if (!t) {
        dbg_head("XML"); std::cerr << "xml_double('" << (path ? path : "") << "') -> <missing>, default=" << def << "\n";
        return def;
    }
    auto txt = t.text();
    if (!txt) {
        dbg_head("XML"); std::cerr << "xml_double('" << (path ? path : "") << "') -> <no-text>, default=" << def << "\n";
        return def;
    }
    double v = txt.as_double(def);
    dbg_head("XML"); std::cerr << "xml_double('" << (path ? path : "") << "') = " << v << "\n";
    return v;
}

const char* xml_str(const pugi::xml_node& n, const char* path, const char* def) {
    auto t = walk_path(n, path);
    if (!t) {
        dbg_head("XML"); std::cerr << "xml_str('" << (path ? path : "") << "') -> <missing>, default='" << (def ? def : "") << "'\n";
        return def;
    }
    auto txt = t.text();
    if (!txt) {
        dbg_head("XML"); std::cerr << "xml_str('" << (path ? path : "") << "') -> <no-text>, default='" << (def ? def : "") << "'\n";
        return def;
    }
    const char* v = txt.as_string(def);
    dbg_head("XML"); std::cerr << "xml_str('" << (path ? path : "") << "') = '" << (v ? v : "") << "'\n";
    return v ? v : def;
}

// -----------------------------------------------------------------------------
// Debug dumper for ISMRMRD header (safe to call; best-effort). Only prints.
// -----------------------------------------------------------------------------
#if ENGINE_HAS_ISMRMRD
bool dump_ismrmrd_metadata(const std::string& path) {
    dbg_head("Meta"); std::cerr << "dump_ismrmrd_metadata path='" << path << "'\n";

    try {
        // Read XML header from ISMRMRD dataset
        ISMRMRD::Dataset d(path.c_str(), "dataset", false);
        std::string xml;
        d.readHeader(xml);

        if (xml.empty()) {
            dbg_head("Meta"); std::cerr << "No XML header present.\n";
            return false;
        }

        // Parse with PugiXML
        pugi::xml_document doc;
        pugi::xml_parse_result ok = doc.load_string(xml.c_str());
        if (!ok) {
            dbg_head("Meta"); std::cerr << "PugiXML parse failed: " << ok.description() << "\n";
            return false;
        }

        auto H = doc.child("ismrmrdHeader");
        if (!H) {
            dbg_head("Meta"); std::cerr << "ismrmrdHeader root not found.\n";
            return false;
        }

        // Print some typical fields
        const char* studyUID = xml_str(H, "studyInformation/studyInstanceUID", "");
        const char* measUID = xml_str(H, "measurementInformation/measurementID", "");
        const char* traj = xml_str(H, "encoding/trajectory", "cartesian");

        dbg_head("Meta"); std::cerr << "studyInstanceUID: " << studyUID << "\n";
        dbg_head("Meta"); std::cerr << "measurementID  : " << measUID << "\n";
        dbg_head("Meta"); std::cerr << "trajectory     : " << traj << "\n";

        // Encoded and recon matrix sizes
        int ex = xml_int(H, "encoding/encodedSpace/matrixSize/x", 0);
        int ey = xml_int(H, "encoding/encodedSpace/matrixSize/y", 0);
        int ez = xml_int(H, "encoding/encodedSpace/matrixSize/z", 0);

        int rx = xml_int(H, "encoding/reconSpace/matrixSize/x", 0);
        int ry = xml_int(H, "encoding/reconSpace/matrixSize/y", 0);
        int rz = xml_int(H, "encoding/reconSpace/matrixSize/z", 0);

        dbg_head("Meta"); std::cerr << "encodedSpace   : [" << ex << "," << ey << "," << ez << "]\n";
        dbg_head("Meta"); std::cerr << "reconSpace     : [" << rx << "," << ry << "," << rz << "]\n";

        // encodingLimits (print common ones if present)
        auto enc = H.child("encoding");
        auto enc_limits = enc.child("encodingLimits");
        auto dump_lim = [enc_limits](const char* tag) {
            auto lim = enc_limits.child(tag);
            if (!lim) return;
            int mn = lim.child("minimum").text().as_int();
            int mx = lim.child("maximum").text().as_int();
            int ct = lim.child("center").text().as_int();
            dbg_head("Meta");
            std::cerr << "  limit(" << tag << "): min=" << mn
                << " max=" << mx << " center=" << ct << "\n";
            };

        dump_lim("kspace_encoding_step_0");
        dump_lim("kspace_encoding_step_1");
        dump_lim("slice");
        dump_lim("average");
        dump_lim("phase");
        dump_lim("repetition");

        dbg_head("Meta"); std::cerr << "dump_ismrmrd_metadata done.\n";
        return true;
    }
    catch (const std::exception& e) {
        dbg_head("Meta"); std::cerr << "Exception: " << e.what() << "\n";
        return false;
    }
    catch (...) {
        dbg_head("Meta"); std::cerr << "Unknown exception.\n";
        return false;
    }
}
#else
bool dump_ismrmrd_metadata(const std::string& path) {
    (void)path;
    dbg_head("Meta"); std::cerr << "dump_ismrmrd_metadata: ISMRMRD disabled at build.\n";
    return false;
}
#endif
