#pragma once
#ifdef _WIN32
  #ifdef MRI_ENGINE_EXPORTS
    #define MRI_API __declspec(dllexport)
  #else
    #define MRI_API __declspec(dllimport)
  #endif
#else
  #define MRI_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
MRI_API int ioismr_init(int device_id);
MRI_API int ioismr_reconstruct_all(const char* path, int* outS, int* outH, int* outW, float** outStack, int fftshift, char* dbg, int dbg_cap);
MRI_API void ioismr_free(void* p);
#ifdef __cplusplus
}
#endif
