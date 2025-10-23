#pragma once

// ===== Export macro (Windows/Linux) =========================================
#if defined(_WIN32) || defined(_WIN64)
#ifdef ENGINE_BUILD
#define ENGINE_API __declspec(dllexport)
#else
#define ENGINE_API __declspec(dllimport)
#endif
#else
#define ENGINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // ---- Version string (overrideable at build time) ---------------------------
    // You can override this in your build:
    //   -DENGINE_VERSION_STR="\"MRI Engine Version 1.1\""
#ifndef ENGINE_VERSION_STR
#define ENGINE_VERSION_STR "MRI Engine Version 1.1"
#endif

// Returns a pointer to a static NUL-terminated string. Do NOT free.
// Thread-safe, lifetime = process.
    ENGINE_API const char* engine_version(void);

    // ============= Public API ====================================================
    ENGINE_API int  engine_init(int device_id);

    /**
     * Reconstruct all slices from the input (DICOM/ISMRMRD/fastMRI, etc.).
     * outStack: returned as contiguous [S * H * W] float32, row-major.
     * Call engine_free(outStack) when done.
     * Returns non-zero on success.
     *
     * If a progress callback is set (engine_set_progress_cb), the engine will
     * invoke it from time to time with percent in [0,100] and a short stage label.
     */
    ENGINE_API int  engine_reconstruct_all(
        const char* path,
        int* outS, int* outH, int* outW,
        float** outStack,
        int fftshift,
        char* dbg, int dbg_cap);

    /** Free memory returned by engine APIs (e.g., outStack from reconstruct). */
    ENGINE_API void engine_free(void* p);

    // ============= Progress Callback API ========================================

    /**
     * Progress callback signature.
     * - percent: 0..100 (clamped)
     * - stage  : optional short label for UI ("Probe", "Load slice", "Reconstruct", "Done", etc.)
     * - user   : opaque pointer provided when registering the callback
     *
     * Notes:
     * - The callback may be invoked from a worker thread or the calling thread,
     *   depending on your integration. If you update a GUI, post to the GUI thread.
     * - The engine will never call this after you clear it (pass NULL).
     */
    typedef void (*engine_progress_fn)(int percent, const char* stage, void* user);

    /**
     * Set or clear the global progress callback.
     * Pass fn=NULL (and user=NULL) to clear.
     */
    ENGINE_API void engine_set_progress_cb(engine_progress_fn fn, void* user);

#ifdef __cplusplus
} // extern "C"
#endif
