#pragma once

/* ===== Visibility + calling convention =================================== */
#if defined(_WIN32) || defined(_WIN64)
#ifdef ENGINE_BUILD
#define ENGINE_API __declspec(dllexport)
#else
#define ENGINE_API __declspec(dllimport)
#endif
/* Fix the ABI across modules: default to __cdecl on Windows */
#ifndef ENGINE_CALL
#define ENGINE_CALL __cdecl
#endif
#else
#define ENGINE_API  __attribute__((visibility("default")))
#ifndef ENGINE_CALL
#define ENGINE_CALL
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /* ===== Version string (override with -DENGINE_VERSION_STR="\"...\"") ====== */
#ifndef ENGINE_VERSION_STR
#define ENGINE_VERSION_STR "MRI Engine Version 1.1"
#endif


    /* Optional: runtime info about the last engine_init() */
    typedef struct engine_init_info_t {
        int struct_size;        /* sizeof(engine_init_info_t) for ABI safety */

        /* CUDA / GPU info */
        int cuda_compiled;      /* 1 if ENGINE_HAS_CUDA != 0 at build time */
        int has_cuda;           /* 1 if a CUDA device was successfully selected */
        int force_cpu;          /* 1 if CPU-only mode was forced (device_id=-1 or MRI_FORCE_CPU) */
        int cuda_device_index;  /* index passed to cudaSetDevice (or -1) */
        int cuda_cc_major;      /* GPU compute capability major */
        int cuda_cc_minor;      /* GPU compute capability minor */
        double cuda_mem_gb;     /* totalGlobalMem in GB */
        char cuda_name[128];    /* device name, NUL-terminated */

        /* OpenMP / CPU info */
        int omp_enabled;        /* 1 if compiled with _OPENMP */
        int omp_max_threads;    /* omp_get_max_threads() if enabled, else 0 */
    } engine_init_info_t;

    /* Fills *out with info about the last engine_init() call.
       If engine_init() has never been called, returns a zeroed struct. */
    ENGINE_API void ENGINE_CALL engine_get_last_init_info(engine_init_info_t* out);


/* Returns a pointer to a static NUL-terminated string. Do NOT free. */
    ENGINE_API const char* ENGINE_CALL engine_version(void);

    /* ===== Progress callback =================================================== */
    /* percent: 0..100, stage: short label, user: opaque pointer you provided */
    typedef void (ENGINE_CALL* engine_progress_fn)(int percent, const char* stage, void* user);
    /* Set or clear (pass NULL) the global progress callback. */
    ENGINE_API void ENGINE_CALL engine_set_progress_cb(engine_progress_fn fn, void* user);

    /* line: single log line (no trailing \n required), user: opaque pointer */
    typedef void (ENGINE_CALL* engine_log_cb)(const char* line, void* user);
    /* Set or clear (pass NULL) the global log callback. */
    ENGINE_API void ENGINE_CALL engine_set_log_cb(engine_log_cb cb, void* user);

    /* ===== Engine lifecycle + reconstruction ================================== */
    /* device_id: >=0 = prefer that CUDA device, -1 = force CPU path */
    ENGINE_API int  ENGINE_CALL engine_init(int device_id);

    /* Reconstruct all slices into a contiguous [S*H*W] float32 buffer (row-major).
       Caller must free with engine_free(outStack). Returns non-zero on success. */
    ENGINE_API int  ENGINE_CALL engine_reconstruct_all(
        const char* path,
        int* outS, int* outH, int* outW,
        float** outStack,
        int fftshift,
        char* dbg, int dbg_cap);

    /* Frees memory returned by the engine (e.g., outStack). Safe for NULL. */
    ENGINE_API void ENGINE_CALL engine_free(void* p);

#ifdef __cplusplus
} /* extern "C" */
#endif
