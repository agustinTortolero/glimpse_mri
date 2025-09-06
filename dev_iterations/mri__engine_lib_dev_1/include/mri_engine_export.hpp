#pragma once
// mri_engine_export.hpp — Windows-friendly dllexport/dllimport macro

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(MRI_ENGINE_BUILD)
    #define MRI_ENGINE_API __declspec(dllexport)
  #else
    #define MRI_ENGINE_API __declspec(dllimport)
  #endif
#else
  #define MRI_ENGINE_API
#endif
