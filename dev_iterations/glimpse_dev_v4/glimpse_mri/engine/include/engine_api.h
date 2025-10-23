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

    ENGINE_API int  engine_reconstruct_all(
        const char* path,
        int* outS, int* outH, int* outW,
        float** outStack,
        int fftshift,
        char* dbg, int dbg_cap);

    ENGINE_API void engine_free(void* p);

#ifdef __cplusplus
} // extern "C"
#endif
