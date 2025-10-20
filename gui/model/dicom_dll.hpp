#pragma once
#include <QString>
#include <QLibrary>
#include <cstdint>
#include <string>
#include <iostream>

struct DicomDll {
    using PFN_probe       = bool (*)(const char*, int*, int*, int*, char*, int);
    using PFN_read        = bool (*)(const char*, int, uint8_t**, int*, int*, char*, int);
    using PFN_free        = void (*)(void*);
    using PFN_write       = bool (*)(const char*, const uint8_t*, int, int, char*, int);

    // NEW
    using PFN_count       = bool (*)(const char*, int*, char*, int);
    using PFN_info        = bool (*)(const char*, int*, int*, int*, int*, int*, int*, char*, int);
    using PFN_read_all    = bool (*)(const char*, uint8_t**, int*, int*, int*, char*, int);

    QLibrary lib;
    PFN_probe    p_probe = nullptr;
    PFN_read     p_read  = nullptr;
    PFN_free     p_free  = nullptr;
    PFN_write    p_write = nullptr;
    PFN_count    p_count = nullptr;    // NEW
    PFN_info     p_info  = nullptr;    // NEW
    PFN_read_all p_read_all = nullptr; // NEW
    bool loaded = false;

    bool load(const QString& explicitPath = QString());

    // Wrappers (debug prints included)
    bool probe(const std::string& path, int* is_multi, int* w, int* h);
    bool read_gray8(const std::string& path, int frame, uint8_t** out, int* w, int* h);
    bool write_sc_gray8(const std::string& path, const uint8_t* buf, int w, int h);
    void free_buf(void* p) {
        if (!p) return;
        if (!loaded || !p_free) {
            std::cerr << "[DICOM][DLL][WRN] free_buf called but DLL not loaded or p_free=null\n";
            return;
        }
        p_free(p);
    }

    // NEW helpers
    bool count_frames(const std::string& path, int* out_frames) {
        char dbg[512]={0}; if (!loaded || !p_count) return false;
        bool ok = p_count(path.c_str(), out_frames, dbg, (int)sizeof(dbg));
        if (dbg[0]) std::cerr << "[DICOM][Count][DBG] " << dbg << "\n";
        return ok;
    }
    bool info(const std::string& path, int* S, int* W, int* H, int* mono, int* spp, int* bits) {
        char dbg[512]={0}; if (!loaded || !p_info) return false;
        bool ok = p_info(path.c_str(), S, W, H, mono, spp, bits, dbg, (int)sizeof(dbg));
        if (dbg[0]) std::cerr << "[DICOM][Info][DBG] " << dbg << "\n";
        return ok;
    }
    bool read_all_gray8(const std::string& path, uint8_t** stack, int* W, int* H, int* S) {
        char dbg[512]={0}; if (!loaded || !p_read_all) return false;
        bool ok = p_read_all(path.c_str(), stack, W, H, S, dbg, (int)sizeof(dbg));
        if (dbg[0]) std::cerr << "[DICOM][ReadAll][DBG] " << dbg << "\n";
        return ok;
    }


};
