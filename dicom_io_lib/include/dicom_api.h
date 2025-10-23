#pragma once
#include <stdint.h>

#ifdef _WIN32
  #ifdef DICOM_IO_LIB_BUILD
    #define DICOM_API __declspec(dllexport)
  #else
    #define DICOM_API __declspec(dllimport)
  #endif
#else
  #define DICOM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Quick probe: returns true if decodable DICOM. Writes human-readable debug into dbg (NUL-terminated).
DICOM_API bool dicom_probe(const char* path,
                           int* is_multiframe,
                           int* out_w0,
                           int* out_h0,
                           char* dbg, int dbg_cap);

// Read one frame to 8-bit grayscale buffer (owned by DLL; free with dicom_free).
DICOM_API bool dicom_read_gray8(const char* path, int frame_index,
                                uint8_t** out_buf, int* out_w, int* out_h,
                                char* dbg, int dbg_cap);

// Free buffer returned by dicom_read_gray8.
DICOM_API void dicom_free(void* p);

// Write Secondary Capture, 8-bit, MONOCHROME2, Explicit Little-Endian.
DICOM_API bool dicom_write_sc_gray8(const char* path,
                                    const uint8_t* buf, int w, int h,
                                    char* dbg, int dbg_cap);

// Returns the number of frames. False if file can't be opened.
DICOM_API bool dicom_count_frames(const char* path,
                                  int* out_frames,
                                  char* dbg, int dbg_cap);

// Returns basic info in a single call.
DICOM_API bool dicom_info(const char* path,
                          int* out_frames,
                          int* out_w, int* out_h,
                          int* out_is_monochrome,   // 1 or 0
                          int* out_samples_per_pixel,
                          int* out_bits_allocated,
                          char* dbg, int dbg_cap);

// Allocates one big block of size (W*H*S). Each frame is 8-bit grayscale, planar.
// Free with dicom_free().
DICOM_API bool dicom_read_all_gray8(const char* path,
                                    uint8_t** out_stack,
                                    int* out_w, int* out_h, int* out_frames,
                                    char* dbg, int dbg_cap);



#ifdef __cplusplus
} // extern "C"
#endif
