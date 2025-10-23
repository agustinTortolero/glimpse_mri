// engine.cpp — aligned to your headers in /include of mri_engine_v_1_1

// If the build system already defines ENGINE_BUILD, avoid redefinition warning.
#ifndef ENGINE_BUILD
#define ENGINE_BUILD
#endif

#include "engine_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <complex>
#include <algorithm>

#include "common.hpp"             // ::KsGrid (coils, ny, nx, host)
#include "fft_and_rss_cpu.hpp"    // fft_and_rss_cpu(...)

#if !defined(ENGINE_HAS_FASTMRI)
#define ENGINE_HAS_FASTMRI 1
#endif
#if !defined(ENGINE_HAS_ISMRMRD)
#define ENGINE_HAS_ISMRMRD 1
#endif

#if ENGINE_HAS_FASTMRI
#include "io_fastmri.hpp"      // FastMRIInfo, fastmri_probe, fastmri_load_kspace_slice
#endif
#if ENGINE_HAS_ISMRMRD
#include "io_ismrmrd.hpp"      // load_ismrmrd_slice(path, slice, step2, ks)
#include <ismrmrd/dataset.h>   // ISMRMRD::Dataset
#include <pugixml.hpp>         // XML parsing
#endif

#if ENGINE_HAS_CUDA
// Declare your CUDA path if you have it linked
extern bool fft_and_rss_cuda(
    int C, int ny, int nx,
    const std::complex<float>* ks,
    std::vector<float>& out_rss);
#endif

// ============================================================================
// Internal state + helpers
// ============================================================================
namespace {

    static int g_force_cpu = 0;
    static int g_device_id = -1;

    // ---- debug buffer helpers ------------------------------------------------
    static void dbg_line(char* dbg, int cap, const char* line) {
        if (dbg && cap > 0) {
            int cur = (int)std::strlen(dbg);
            int len = (int)std::strlen(line);
            if (cur + len + 2 < cap) {
                std::strcat(dbg, line);
                std::strcat(dbg, "\n");
            }
        }
        std::fprintf(stderr, "%s\n", line);
    }

    template <typename... Args>
    static void dbg_printf(char* dbg, int cap, const char* fmt, Args... args) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        dbg_line(dbg, cap, buf);
    }

    // ---- clarity-first fftshift (copy-and-permute) ---------------------------
    static void fftshift2d_inplace(float* img, int ny, int nx) {
        std::vector<float> tmp((size_t)ny * nx);
        const int cy = ny / 2, cx = nx / 2;
        auto idx = [nx](int y, int x) { return (size_t)y * nx + x; };
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int yy = (y + cy) % ny;
                int xx = (x + cx) % nx;
                tmp[idx(yy, xx)] = img[idx(y, x)];
            }
        }
        std::memcpy(img, tmp.data(), (size_t)ny * nx * sizeof(float));
    }

    // ---- minimal flavor + detection model -----------------------------------
    enum class Flavor : int { Unknown = 0, ISMRMRD = 1, FastMRI = 2 };

    struct DetectedInfo {
        Flavor flavor{ Flavor::Unknown };
        int S{ 0 }, C{ 0 }, ny{ 0 }, nx{ 0 };
        bool has_rss{ false };
    };

    // ---- Progress callback plumbing (lives here so every symbol sees it) -----
    // Header typedef: typedef void (*engine_progress_fn)(int percent, const char* stage, void* user);
    static engine_progress_fn g_prog_cb = nullptr;
    static void* g_prog_ud = nullptr;

    /**
     * Helper to clamp & emit progress:
     * - Writes to dbg (if provided) and stderr for traceability.
     * - Calls the user callback if set.
     */
    static inline void report_progress(int pct, const char* stage, char* dbg = nullptr, int cap = 0)
    {
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;

        const char* tag = stage ? stage : "";

        if (dbg) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "[DBG][ENGINE][PROG] %3d%% %s", pct, tag);
            dbg_line(dbg, cap, buf);
        }
        else {
            std::fprintf(stderr, "[DBG][ENGINE][PROG] %3d%% %s\n", pct, tag);
        }

        if (g_prog_cb) {
            g_prog_cb(pct, tag, g_prog_ud);
        }
    }

    // ---------- ISMRMRD slice-range probe ------------------------------------
#if ENGINE_HAS_ISMRMRD
    static bool ismrmrd_get_slice_limits(const std::string& path, int& smin, int& smax, char* dbg, int cap) {
        dbg_printf(dbg, cap, "[DBG][ISMR] get_slice_limits '%s'", path.c_str());
        try {
            ISMRMRD::Dataset d(path.c_str(), "dataset", false);
            std::string xml; d.readHeader(xml);
            pugi::xml_document doc;
            if (!doc.load_string(xml.c_str())) {
                dbg_line(dbg, cap, "[DBG][ISMR] XML parse failed");
                return false;
            }
            auto lim = doc.child("ismrmrdHeader")
                .child("encoding")
                .child("encodingLimits")
                .child("slice");
            smin = lim.child("minimum").text().as_int(0);
            smax = lim.child("maximum").text().as_int(0);
            if (smax < smin) std::swap(smin, smax);
            dbg_printf(dbg, cap, "[DBG][ISMR] slice limits: [%d..%d]", smin, smax);
            return true;
        }
        catch (...) {
            dbg_line(dbg, cap, "[DBG][ISMR] exception reading ISMRMRD header");
            return false;
        }
    }
#endif

    // ---------- Probing -------------------------------------------------------
    static bool detect_fastmri(const std::string& path, DetectedInfo& out, char* dbg, int cap) {
#if ENGINE_HAS_FASTMRI
        FastMRIInfo info{};
        if (!fastmri_probe(path, info)) {
            dbg_line(dbg, cap, "[DBG][ENGINE] fastMRI probe: not fastMRI");
            return false;
        }
        out.flavor = Flavor::FastMRI;
        out.S = info.slices;
        out.C = info.coils;
        out.ny = info.ny;
        out.nx = info.nx;
        out.has_rss = info.has_rss;
        dbg_printf(dbg, cap, "[DBG][ENGINE] fastMRI detected: S=%d C=%d ny=%d nx=%d has_rss=%d",
            out.S, out.C, out.ny, out.nx, out.has_rss ? 1 : 0);
        return true;
#else
        (void)path; (void)out; dbg_line(dbg, cap, "[DBG][ENGINE] fastMRI disabled at build");
        return false;
#endif
    }

    static bool detect_ismrmrd(const std::string& path, DetectedInfo& out, char* dbg, int cap) {
#if ENGINE_HAS_ISMRMRD
        int smin = 0, smax = -1;
        if (!ismrmrd_get_slice_limits(path, smin, smax, dbg, cap)) {
            dbg_line(dbg, cap, "[DBG][ENGINE] ISMRMRD probe: not ISMRMRD");
            return false;
        }
        const int Z = smax - smin + 1;
        if (Z <= 0) {
            dbg_line(dbg, cap, "[DBG][ENGINE] ISMRMRD: invalid slice range");
            return false;
        }

        // Load first slice to get C/ny/nx (matches your legacy approach)
        ::KsGrid ks0;
        if (!load_ismrmrd_slice(path, smin, /*step2=*/-1, ks0)) {
            dbg_line(dbg, cap, "[DBG][ENGINE] ISMRMRD: load first slice failed");
            return false;
        }

        out.flavor = Flavor::ISMRMRD;
        out.S = Z;
        out.C = ks0.coils;
        out.ny = ks0.ny;
        out.nx = ks0.nx;
        out.has_rss = false;

        dbg_printf(dbg, cap, "[DBG][ENGINE] ISMRMRD detected: S=%d C=%d ny=%d nx=%d range=[%d..%d]",
            out.S, out.C, out.ny, out.nx, smin, smax);
        return true;
#else
        (void)path; (void)out; dbg_line(dbg, cap, "[DBG][ENGINE] ISMRMRD disabled at build");
        return false;
#endif
    }

    static bool detect_flavor(const std::string& path, DetectedInfo& out, char* dbg, int cap) {
        if (detect_fastmri(path, out, dbg, cap)) return true;
        if (detect_ismrmrd(path, out, dbg, cap)) return true;
        return false;
    }

    // ---------- Slice loading (into ::KsGrid) ---------------------------------
    static bool load_slice_fastmri(const std::string& path, int s, ::KsGrid& ks, char* dbg, int cap) {
#if ENGINE_HAS_FASTMRI
        if (!fastmri_load_kspace_slice(path, s, ks)) {
            dbg_printf(dbg, cap, "[ERR][ENGINE] fastMRI load slice %d failed", s);
            return false;
        }
        dbg_printf(dbg, cap, "[DBG][ENGINE] fastMRI slice %d loaded  coils=%d ny=%d nx=%d",
            s, ks.coils, ks.ny, ks.nx);
        return true;
#else
        (void)path; (void)s; (void)ks; dbg_line(dbg, cap, "[ERR][ENGINE] fastMRI backend disabled");
        return false;
#endif
    }

    static bool load_slice_ismrmrd(const std::string& path, int s, ::KsGrid& ks, char* dbg, int cap) {
#if ENGINE_HAS_ISMRMRD
        // NOTE: your header requires (path, slice_idx, step2, KsGrid&)
        if (!load_ismrmrd_slice(path, s, /*step2=*/-1, ks)) {
            dbg_printf(dbg, cap, "[ERR][ENGINE] ISMRMRD load slice %d failed", s);
            return false;
        }
        dbg_printf(dbg, cap, "[DBG][ENGINE] ISMRMRD slice %d loaded  coils=%d ny=%d nx=%d",
            s, ks.coils, ks.ny, ks.nx);
        return true;
#else
        (void)path; (void)s; (void)ks; dbg_line(dbg, cap, "[ERR][ENGINE] ISMRMRD backend disabled");
        return false;
#endif
    }

    // ---------- Reconstruction (CUDA optional, CPU fallback) ------------------
    static bool reconstruct_slice_rss(const ::KsGrid& ks, std::vector<float>& out_img, bool prefer_cuda, char* dbg, int cap) {
        out_img.clear();
        out_img.resize((size_t)ks.ny * ks.nx);

#if ENGINE_HAS_CUDA
        if (prefer_cuda) {
            dbg_line(dbg, cap, "[DBG][ENGINE] trying CUDA backend");
            if (fft_and_rss_cuda(ks.coils, ks.ny, ks.nx, ks.host.data(), out_img)) {
                dbg_line(dbg, cap, "[DBG][ENGINE] CUDA backend success");
                return true;
            }
            dbg_line(dbg, cap, "[DBG][ENGINE] CUDA backend failed -> falling back to CPU");
        }
        else {
            dbg_line(dbg, cap, "[DBG][ENGINE] CPU-only mode (user forced or env)");
        }
#else
        (void)prefer_cuda;
        dbg_line(dbg, cap, "[DBG][ENGINE] CUDA backend not compiled -> CPU path");
#endif

        // CPU fallback
        bool ok = fft_and_rss_cpu(ks.coils, ks.ny, ks.nx, ks.host.data(), out_img);
        if (!ok) {
            dbg_line(dbg, cap, "[ERR][ENGINE] CPU reconstruction failed");
            return false;
        }
        dbg_line(dbg, cap, "[DBG][ENGINE] CPU backend success");
        return true;
    }

} // namespace

// ============================================================================
// C API
// ============================================================================
extern "C" {

    ENGINE_API const char* engine_version(void) {
        static const char* kVersion = ENGINE_VERSION_STR;
        std::fprintf(stderr, "[DBG][ENGINE] engine_version -> %s\n", kVersion);
        return kVersion;
    }

    ENGINE_API int engine_init(int device_id) {
        g_device_id = device_id;

        const char* env_force = std::getenv("MRI_FORCE_CPU");
        g_force_cpu = (device_id == -1) || (env_force && std::strlen(env_force) > 0);

        std::fprintf(stderr,
            "[DBG][ENGINE] engine_init(device_id=%d) MRI_FORCE_CPU=%s -> force_cpu=%s\n",
            device_id,
            env_force ? env_force : "null",
            g_force_cpu ? "true" : "false");

        return 1; // success
    }

    // NEW: Set/Clear the global progress callback (pairs with header)
    ENGINE_API void engine_set_progress_cb(engine_progress_fn fn, void* user)
    {
        g_prog_cb = fn;
        g_prog_ud = user;
        std::fprintf(stderr, "[DBG][ENGINE] engine_set_progress_cb: fn=%p user=%p\n",
            (void*)fn, user);
    }

    ENGINE_API int engine_reconstruct_all(
        const char* path,
        int* outS, int* outH, int* outW,
        float** outStack,
        int fftshift,
        char* dbg, int dbg_cap)
    {
        // ------------------- argument checks -------------------
        if (!path || !outS || !outH || !outW || !outStack) {
            dbg_line(dbg, dbg_cap, "[ERR][ENGINE] invalid null argument(s)");
            report_progress(100, "Error: args", dbg, dbg_cap);
            return 0;
        }

        *outS = *outH = *outW = 0;
        *outStack = nullptr;

        report_progress(1, "Start", dbg, dbg_cap);
        dbg_printf(dbg, dbg_cap,
            "[DBG][ENGINE] reconstruct_all path=\"%s\" fftshift=%d backend=%s",
            path,
            fftshift ? 1 : 0,
            g_force_cpu ? "CPU" : "AUTO");

        // ------------------- probe / detect flavor -------------------
        DetectedInfo di{};
        if (!detect_flavor(path, di, dbg, dbg_cap)) {
            dbg_line(dbg, dbg_cap, "[ERR][ENGINE] unknown file flavor (not fastMRI/ISMRMRD)");
            report_progress(100, "Error: flavor", dbg, dbg_cap);
            return 0;
        }

        if (di.S <= 0 || di.C <= 0 || di.ny <= 0 || di.nx <= 0) {
            dbg_printf(dbg, dbg_cap, "[ERR][ENGINE] invalid dims after probe S=%d C=%d ny=%d nx=%d",
                di.S, di.C, di.ny, di.nx);
            report_progress(100, "Error: dims", dbg, dbg_cap);
            return 0;
        }

        report_progress(5, "Probe/Dimensions", dbg, dbg_cap);

        // ------------------- allocate output stack -------------------
        const size_t imgN = (size_t)di.ny * (size_t)di.nx;
        const size_t stackN = (size_t)di.S * imgN;

        float* stack = (float*)std::malloc(stackN * sizeof(float));
        if (!stack) {
            dbg_line(dbg, dbg_cap, "[ERR][ENGINE] malloc failed for output stack");
            report_progress(100, "Error: alloc", dbg, dbg_cap);
            return 0;
        }

        report_progress(8, "Alloc output", dbg, dbg_cap);

        // ------------------- process each slice -------------------
        std::vector<float> slice_img;
        slice_img.reserve(imgN);

        // reserve 10% pre/post, distribute 80% across slices
        const int base = 10;
        const int span = 80;
        const int denom = (di.S > 0) ? di.S : 1;

        for (int s = 0; s < di.S; ++s) {
            ::KsGrid ks; // from your common.hpp / loader path
            bool ok_load = false;

            // progress: loading stage
            {
                const int pct = base + (s * span) / denom;
                report_progress(pct, "Load slice", dbg, dbg_cap);
            }

            if (di.flavor == Flavor::FastMRI) {
                ok_load = load_slice_fastmri(path, s, ks, dbg, dbg_cap);
            }
            else { // Flavor::ISMRMRD
                ok_load = load_slice_ismrmrd(path, s, ks, dbg, dbg_cap);
            }

            if (!ok_load) {
                dbg_printf(dbg, dbg_cap, "[ERR][ENGINE] failed loading slice %d", s);
                std::free(stack);
                report_progress(100, "Error: load", dbg, dbg_cap);
                return 0;
            }

            // progress: reconstruct stage (middle of this slice share)
            {
                const int pct = base + ((s * span) + span / 2) / denom;
                report_progress(pct, "Reconstruct", dbg, dbg_cap);
            }

            const bool prefer_cuda = !g_force_cpu;
            if (!reconstruct_slice_rss(ks, slice_img, prefer_cuda, dbg, dbg_cap)) {
                dbg_printf(dbg, dbg_cap, "[ERR][ENGINE] reconstruction failed on slice %d", s);
                std::free(stack);
                report_progress(100, "Error: recon", dbg, dbg_cap);
                return 0;
            }

            if (fftshift) {
                dbg_printf(dbg, dbg_cap, "[DBG][ENGINE][Shift] fftshift2d_inplace ny=%d nx=%d", ks.ny, ks.nx);
                fftshift2d_inplace(slice_img.data(), ks.ny, ks.nx);
            }

            // copy result into the stack
            std::memcpy(stack + (size_t)s * imgN, slice_img.data(), imgN * sizeof(float));

            // progress: packed stage
            {
                const int pct = base + ((s + 1) * span) / denom;
                report_progress(pct, "Pack", dbg, dbg_cap);
            }

            // occasional textual progress
            if (((s + 1) % 8) == 0 || s == di.S - 1) {
                dbg_printf(dbg, dbg_cap, "[DBG][ENGINE] progress: slice %d/%d", s + 1, di.S);
            }
        }

        // ------------------- publish -------------------
        report_progress(98, "Finalize", dbg, dbg_cap);

        *outS = di.S;
        *outH = di.ny;
        *outW = di.nx;
        *outStack = stack;

        // legacy-compat return (1=ISMRMRD, 2=fastMRI)
        const int rc = (di.flavor == Flavor::ISMRMRD) ? 1 : 2;

        report_progress(100, "Done", dbg, dbg_cap);
        return rc;
    }

    ENGINE_API void engine_free(void* p) {
        if (!p) {
            std::fprintf(stderr, "[DBG][ENGINE] engine_free(nullptr) — no-op\n");
            return;
        }
        std::fprintf(stderr, "[DBG][ENGINE] engine_free(ptr)\n");
        std::free(p);
    }

} // extern "C"
