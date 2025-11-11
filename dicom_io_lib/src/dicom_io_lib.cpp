// dicom_io_lib/src/dicom_io_lib.cpp
// Keep it simple: Monochrome DICOM → 8-bit output. Basic write (SC MONOCHROME2).

#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <string>
#include <new>          // std::nothrow
#include <exception>    // std::exception
#include <limits>       // std::numeric_limits  [Linux/GCC/Clang friendly]

// ---- DCMTK: osconfig FIRST ----
#include <dcmtk/config/osconfig.h>

// ---- DCMTK core ----
#include <dcmtk/dcmimgle/dcmimage.h>   // DicomImage
#include <dcmtk/dcmdata/dctk.h>        // DcmFileFormat, DcmDataset
#include <dcmtk/dcmdata/dcdeftag.h>    // tags
#include <dcmtk/dcmdata/dcuid.h>       // UIDs

// ---- our public C API ----
#include "../include/dicom_api.h"

// ----------------------- debug helper (append) -----------------------
static void dbg_put(char* dbg, int cap, const char* fmt, ...) {
    if (!dbg || cap <= 0) return;
    size_t used = strnlen(dbg, (size_t)cap);
    if (used >= (size_t)cap) return;
    va_list ap; va_start(ap, fmt);
#if defined(_WIN32)
    _vsnprintf_s(dbg + used, cap - (int)used, _TRUNCATE, fmt, ap);
#else
    vsnprintf(dbg + used, (size_t)cap - used, fmt, ap);
#endif
    va_end(ap);
}

// -------------------------- small utilities --------------------------
namespace {

constexpr size_t kMaxStackBytes = 1024ull * 1024ull * 1024ull; // 1 GiB guard

inline bool valid_path(const char* p) { return p && *p; }

inline size_t safe_plane(int w, int h) {
    if (w <= 0 || h <= 0) return 0;
    // use 64-bit math for safety, clamp to size_t on return
    const unsigned long long P = (unsigned long long)w * (unsigned long long)h;
    if (P > (unsigned long long)std::numeric_limits<size_t>::max()) return 0;
    return (size_t)P;
}

inline size_t safe_stack_bytes(int w, int h, int s) {
    if (s <= 0) return 0;
    const size_t plane = safe_plane(w, h);
    if (plane == 0) return 0;
    // 64-bit multiply
    const unsigned long long B = (unsigned long long)plane * (unsigned long long)s;
    if (B > (unsigned long long)std::numeric_limits<size_t>::max()) return 0;
    return (size_t)B;
}

inline size_t checksum16(const uint8_t* src, size_t n) {
    if (!src || n == 0) return 0;
    const size_t step = n / 16 + 1; // clarity > perf
    size_t cs = 0;
    for (size_t i = 0; i < n; i += step) cs += src[i];
    return cs;
}

} // anonymous namespace

// ====================================================================
// C API
// ====================================================================
extern "C" {

// --------------------------------------------------------------------
// PROBE / INFO
// --------------------------------------------------------------------
bool dicom_probe(const char* path, int* is_multiframe, int* out_w0, int* out_h0,
                 char* dbg, int dbg_cap)
{
    try {
        dbg_put(dbg, dbg_cap, "[DICOM][Probe] '%s'\n", path ? path : "(null)");
        if (!valid_path(path)) { dbg_put(dbg, dbg_cap, "[ERR] null/empty path\n"); return false; }

        // NOTE: DCMTK’s DicomImage(char const*) expects local-encoding path.
        // For Unicode-only Windows paths consider using OFFilename + UTF-8 conversion.
        DicomImage di(path);
        if (di.getStatus() != EIS_Normal) {
            dbg_put(dbg, dbg_cap, "[ERR] DicomImage status=%d\n", (int)di.getStatus());
            return false;
        }

        if (is_multiframe) *is_multiframe = (di.getFrameCount() > 1) ? 1 : 0;
        if (out_w0) *out_w0 = (int)di.getWidth();
        if (out_h0) *out_h0 = (int)di.getHeight();

        dbg_put(dbg, dbg_cap, "[OK ] %dx%d frames=%lu mono=%d\n",
                (int)di.getWidth(), (int)di.getHeight(),
                (unsigned long)di.getFrameCount(),
                di.isMonochrome() ? 1 : 0);
        return true;
    } catch (const std::exception& e) {
        dbg_put(dbg, dbg_cap, "[EXC] %s\n", e.what());
        return false;
    } catch (...) {
        dbg_put(dbg, dbg_cap, "[EXC] unknown\n");
        return false;
    }
}

bool dicom_info(const char* path,
                int* out_frames,
                int* out_w, int* out_h,
                int* out_is_monochrome,
                int* out_samples_per_pixel,
                int* out_bits_allocated,
                char* dbg, int dbg_cap)
{
    try {
        dbg_put(dbg, dbg_cap, "[DICOM][Info] '%s'\n", path ? path : "(null)");
        if (!valid_path(path)) { dbg_put(dbg, dbg_cap, "[ERR] bad args\n"); return false; }

        DicomImage di(path);
        if (di.getStatus() != EIS_Normal) {
            dbg_put(dbg, dbg_cap, "[ERR] DicomImage status=%d\n", (int)di.getStatus());
            return false;
        }

        if (out_w) *out_w = (int)di.getWidth();
        if (out_h) *out_h = (int)di.getHeight();
        if (out_frames) *out_frames = (int)di.getFrameCount();
        if (out_is_monochrome) *out_is_monochrome = di.isMonochrome() ? 1 : 0;

        // Peek SPP/bits via dataset (best effort)
        int spp = di.isMonochrome() ? 1 : 3;
        int bits = 8;
        DcmFileFormat ff;
        if (ff.loadFile(path).good()) {
            DcmDataset* ds = ff.getDataset();
            Uint16 u16 = 0;
            if (ds && ds->findAndGetUint16(DCM_SamplesPerPixel, u16).good()) spp = (int)u16;
            if (ds && ds->findAndGetUint16(DCM_BitsAllocated, u16).good())   bits = (int)u16;
        }
        if (out_samples_per_pixel) *out_samples_per_pixel = spp;
        if (out_bits_allocated)    *out_bits_allocated    = bits;

        dbg_put(dbg, dbg_cap, "[OK ] %dx%d frames=%d mono=%d spp=%d bits=%d\n",
                out_w?*out_w:0, out_h?*out_h:0, out_frames?*out_frames:0,
                out_is_monochrome?*out_is_monochrome:0, spp, bits);
        return true;
    } catch (const std::exception& e) {
        dbg_put(dbg, dbg_cap, "[EXC] %s\n", e.what());
        return false;
    } catch (...) {
        dbg_put(dbg, dbg_cap, "[EXC] unknown\n");
        return false;
    }
}

bool dicom_count_frames(const char* path, int* out_frames, char* dbg, int dbg_cap)
{
    try {
        dbg_put(dbg, dbg_cap, "[DICOM][Count] '%s'\n", path ? path : "(null)");
        if (!valid_path(path) || !out_frames) { dbg_put(dbg, dbg_cap, "[ERR] bad args\n"); return false; }
        DicomImage di(path);
        if (di.getStatus() != EIS_Normal) {
            dbg_put(dbg, dbg_cap, "[ERR] DicomImage status=%d\n", (int)di.getStatus());
            return false;
        }
        *out_frames = (int)di.getFrameCount();
        dbg_put(dbg, dbg_cap, "[OK ] frames=%d\n", *out_frames);
        return true;
    } catch (const std::exception& e) {
        dbg_put(dbg, dbg_cap, "[EXC] %s\n", e.what());
        return false;
    } catch (...) {
        dbg_put(dbg, dbg_cap, "[EXC] unknown\n");
        return false;
    }
}

// --------------------------------------------------------------------
// READ ONE FRAME (8-bit, MONOCHROME only)
// --------------------------------------------------------------------
bool dicom_read_gray8(const char* path, int frame_index,
                      uint8_t** out_buf, int* out_w, int* out_h,
                      char* dbg, int dbg_cap)
{
    try {
        dbg_put(dbg, dbg_cap, "[DICOM][Read] path='%s' frame=%d\n", path ? path : "(null)", frame_index);
        if (!valid_path(path) || !out_buf || !out_w || !out_h) {
            dbg_put(dbg, dbg_cap, "[ERR] bad args\n"); return false;
        }
        *out_buf = nullptr; *out_w = 0; *out_h = 0;

        DicomImage di(path);
        if (di.getStatus() != EIS_Normal) {
            dbg_put(dbg, dbg_cap, "[ERR] DicomImage status=%d\n", int(di.getStatus()));
            return false;
        }

        // admission: monochrome only
        if (!di.isMonochrome()) {
            dbg_put(dbg, dbg_cap, "[ERR] not admitted: requires MONOCHROME Photometric\n");
            return false;
        }
        di.setMinMaxWindow(); // window to 8-bit nicely

        const int frames = int(di.getFrameCount());
        if (frames <= 0) { dbg_put(dbg, dbg_cap, "[ERR] no frames\n"); return false; }
        if (frame_index < 0 || frame_index >= frames) {
            dbg_put(dbg, dbg_cap, "[WRN] clamping frame %d->0\n", frame_index);
            frame_index = 0;
        }

        const int W = int(di.getWidth());
        const int H = int(di.getHeight());
        const size_t plane = safe_plane(W, H);
        if (plane == 0) { dbg_put(dbg, dbg_cap, "[ERR] invalid dims %dx%d\n", W, H); return false; }

        // (bits, frame, planar)
        const void* p = di.getOutputData(8 /*bits*/, (unsigned long)frame_index /*frame*/, 0 /*planar*/);
        if (!p) { dbg_put(dbg, dbg_cap, "[ERR] getOutputData(8,%d,0) null\n", frame_index); return false; }

        uint8_t* out = static_cast<uint8_t*>(::operator new(plane, std::nothrow));
        if (!out) { dbg_put(dbg, dbg_cap, "[ERR] alloc %zu bytes failed\n", plane); return false; }
        std::memcpy(out, p, plane);

        // Debug fingerprint
        const auto* u = static_cast<const uint8_t*>(p);
        const uint8_t s0 = u[0];
        const uint8_t sm = u[plane / 2];
        const uint8_t sl = u[plane - 1];
        const size_t cs = checksum16(u, plane);

        *out_buf = out;
        *out_w   = W;
        *out_h   = H;

        dbg_put(dbg, dbg_cap,
                "[OK ] read -> %dx%d (8-bit) samples={first:%u mid:%u last:%u} checksum=%zu\n",
                W, H, (unsigned)s0, (unsigned)sm, (unsigned)sl, cs);
        return true;
    } catch (const std::exception& e) {
        dbg_put(dbg, dbg_cap, "[EXC] %s\n", e.what());
        return false;
    } catch (...) {
        dbg_put(dbg, dbg_cap, "[EXC] unknown\n");
        return false;
    }
}

// --------------------------------------------------------------------
// READ ALL FRAMES (contiguous 8-bit stack, MONOCHROME only)
// --------------------------------------------------------------------
bool dicom_read_all_gray8(const char* path,
                          uint8_t** out_stack,
                          int* out_w, int* out_h, int* out_frames,
                          char* dbg, int dbg_cap)
{
    try {
        dbg_put(dbg, dbg_cap, "[DICOM][ReadAll] path='%s'\n", path ? path : "(null)");
        if (!valid_path(path) || !out_stack || !out_w || !out_h || !out_frames) {
            dbg_put(dbg, dbg_cap, "[ERR] bad args\n");
            return false;
        }
        *out_stack = nullptr; *out_w = 0; *out_h = 0; *out_frames = 0;

        DicomImage di(path);
        if (di.getStatus() != EIS_Normal) {
            dbg_put(dbg, dbg_cap, "[ERR] DicomImage status=%d\n", int(di.getStatus()));
            return false;
        }

        // admission: monochrome only
        if (!di.isMonochrome()) {
            dbg_put(dbg, dbg_cap, "[ERR] not admitted: requires MONOCHROME Photometric\n");
            return false;
        }
        di.setMinMaxWindow(); // window to 8-bit nicely

        const int W = int(di.getWidth());
        const int H = int(di.getHeight());
        const int S = int(di.getFrameCount());
        if (W <= 0 || H <= 0 || S <= 0) {
            dbg_put(dbg, dbg_cap, "[ERR] invalid dims=%dx%d frames=%d\n", W, H, S);
            return false;
        }

        const size_t plane = safe_plane(W, H);
        const size_t bytes = safe_stack_bytes(W, H, S);
        if (plane == 0 || bytes == 0) {
            dbg_put(dbg, dbg_cap, "[ERR] invalid/overflow plane or bytes (W=%d H=%d S=%d)\n", W, H, S);
            return false;
        }
        if (bytes > kMaxStackBytes) {
            dbg_put(dbg, dbg_cap, "[ERR] stack too large (%zu bytes > %zu limit)\n", bytes, kMaxStackBytes);
            return false;
        }

        uint8_t* stack = static_cast<uint8_t*>(::operator new(bytes, std::nothrow));
        if (!stack) { dbg_put(dbg, dbg_cap, "[ERR] alloc %zu bytes failed\n", bytes); return false; }

        for (int f = 0; f < S; ++f) {
            const void* p = di.getOutputData(8 /*bits*/, (unsigned long)f /*frame*/, 0 /*planar*/);
            if (!p) {
                dbg_put(dbg, dbg_cap, "[ERR] frame %d: getOutputData(8,frame,0) null\n", f);
                ::operator delete(stack);
                return false;
            }
            // Copy out this frame
            std::memcpy(stack + plane * (size_t)f, p, plane);

            // Debug fingerprint
            const auto* u = static_cast<const uint8_t*>(p);
            const uint8_t s0 = u[0];
            const uint8_t sm = u[plane / 2];
            const uint8_t sl = u[plane - 1];
            const size_t cs = checksum16(u, plane);

            dbg_put(dbg, dbg_cap,
                    "[OK ] frame %d/%d -> %dx%d  samples={first:%u mid:%u last:%u} checksum=%zu\n",
                    f, S, W, H, (unsigned)s0, (unsigned)sm, (unsigned)sl, cs);
        }

        *out_stack  = stack;
        *out_w      = W;
        *out_h      = H;
        *out_frames = S;
        dbg_put(dbg, dbg_cap, "[OK ] read-all -> %dx%d x %d frames (%zu bytes)\n", W, H, S, bytes);
        return true;
    } catch (const std::exception& e) {
        dbg_put(dbg, dbg_cap, "[EXC] %s\n", e.what());
        return false;
    } catch (...) {
        dbg_put(dbg, dbg_cap, "[EXC] unknown\n");
        return false;
    }
}

// --------------------------------------------------------------------
// FREE
// --------------------------------------------------------------------
void dicom_free(void* p) { ::operator delete(p); }

// --------------------------------------------------------------------
// WRITE (SC MONOCHROME2, 8-bit)
// --------------------------------------------------------------------
bool dicom_write_sc_gray8(const char* path, const uint8_t* buf, int w, int h,
                          char* dbg, int dbg_cap)
{
    try {
        dbg_put(dbg, dbg_cap, "[DICOM][Write] '%s' dims=%dx%d\n", path ? path : "(null)", w, h);
        if (!valid_path(path) || !buf || w <= 0 || h <= 0) {
            dbg_put(dbg, dbg_cap, "[ERR] bad args\n"); return false;
        }

        DcmFileFormat ff;
        DcmDataset* ds = ff.getDataset();

        // UIDs
        char studyUID[64]={0}, seriesUID[64]={0}, instUID[64]={0};
        dcmGenerateUniqueIdentifier(studyUID);
        dcmGenerateUniqueIdentifier(seriesUID);
        dcmGenerateUniqueIdentifier(instUID);

        ds->putAndInsertString(DCM_SOPClassUID,          UID_SecondaryCaptureImageStorage);
        ds->putAndInsertString(DCM_SOPInstanceUID,       instUID);
        ds->putAndInsertString(DCM_StudyInstanceUID,     studyUID);
        ds->putAndInsertString(DCM_SeriesInstanceUID,    seriesUID);
        ds->putAndInsertString(DCM_Modality,             "SC");
        ds->putAndInsertString(DCM_SpecificCharacterSet, "ISO_IR 192"); // UTF-8

        ds->putAndInsertUint16(DCM_Rows,                      (Uint16)h);
        ds->putAndInsertUint16(DCM_Columns,                   (Uint16)w);
        ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
        ds->putAndInsertUint16(DCM_SamplesPerPixel,           1);
        ds->putAndInsertUint16(DCM_BitsAllocated,             8);
        ds->putAndInsertUint16(DCM_BitsStored,                8);
        ds->putAndInsertUint16(DCM_HighBit,                   7);
        ds->putAndInsertUint16(DCM_PixelRepresentation,       0);

        const Uint32 nbytes = (Uint32)((size_t)w * (size_t)h);
        OFCondition st = ds->putAndInsertUint8Array(DCM_PixelData, (const Uint8*)buf, nbytes);
        if (st.bad()) { dbg_put(dbg, dbg_cap, "[ERR] PixelData: %s\n", st.text()); return false; }

        OFCondition saveSt = ff.saveFile(path, EXS_LittleEndianExplicit);
        if (saveSt.bad()) { dbg_put(dbg, dbg_cap, "[ERR] saveFile: %s\n", saveSt.text()); return false; }

        dbg_put(dbg, dbg_cap, "[OK ] wrote SC MONOCHROME2 8-bit\n");
        return true;
    } catch (const std::exception& e) {
        dbg_put(dbg, dbg_cap, "[EXC] %s\n", e.what());
        return false;
    } catch (...) {
        dbg_put(dbg, dbg_cap, "[EXC] unknown\n");
        return false;
    }
}

} // extern "C"
