# =========================
# mri__engine_lib.pro  (Windows-only; pre-link NVCC; CUDA short path)
# =========================

TEMPLATE = lib
TARGET   = mri__engine_lib
CONFIG  += c++17 dll shared debug_and_release
CONFIG  += warn_on
CONFIG  += skip_target_version_ext

# ---------- Relative build dirs ----------
OBJECTS_DIR = obj
MOC_DIR     = moc
RCC_DIR     = rcc
UI_DIR      = ui

# ---------- Sources / headers ----------
INCLUDEPATH += $$PWD/include
VPATH       += $$PWD/src

HEADERS += \
    $$PWD/include/mri_engine_export.hpp \
    $$PWD/include/mri_engine_api.h \
    $$PWD/include/mri_engine.hpp

# Only C++ sources here (DO NOT list .cu here)
SOURCES += \
    $$PWD/src/api_exports.cpp

# CUDA source we compile via NVCC in pre-link
CUDA_SOURCES = $$PWD/src/mri_engine.cu

# ---------- (optional) vcpkg headers ----------
!isEmpty($$(VCPKG_ROOT)) {
    VCPKG_ROOT = $$(VCPKG_ROOT)
} else {
    exists("C:/src/vcpkg/installed/x64-windows/include") {
        VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
    } else: exists("C:/AgustinTortolero_repos/MRI/vcpkg/installed/x64-windows/include") {
        VCPKG_ROOT = C:/AgustinTortolero_repos/MRI/vcpkg/installed/x64-windows
    } else {
        VCPKG_ROOT =
    }
}
!isEmpty(VCPKG_ROOT) {
    message([DBG] Using VCPKG_ROOT = $$VCPKG_ROOT)
    INCLUDEPATH += $$VCPKG_ROOT/include
}

# =========================
# CUDA paths (SHORT 8.3 path to avoid spaces)
# =========================
# Use the 8.3 path you printed from cmd. Adjust if yours differs:
CUDA_PATH = C:/PROGRA~1/NVIDIA~2/CUDA/v12.4
message([DBG] Using CUDA_PATH = $$CUDA_PATH)

# Compute capability
isEmpty(CUDA_ARCH) { CUDA_ARCH = 89 }
message([DBG] NVCC arch = sm_$$CUDA_ARCH)
message([DBG] CUDA_SOURCES = $$CUDA_SOURCES)

# =========================
# Include + Libs (no spaces thanks to short path)
# =========================
CUDA_INC_DIR = $$CUDA_PATH/include
INCLUDEPATH += $$CUDA_INC_DIR
message([DBG] CUDA_INC_DIR = $$CUDA_INC_DIR)

CUDA_LIB_DIR = $$CUDA_PATH/lib/x64
QMAKE_LIBDIR += $$CUDA_LIB_DIR
message([DBG] CUDA_LIB_DIR = $$CUDA_LIB_DIR)

# Link the CUDA libs you have
LIBS += $$CUDA_LIB_DIR/cudart.lib
LIBS += $$CUDA_LIB_DIR/cufft.lib
LIBS += $$CUDA_LIB_DIR/cublas.lib
LIBS += $$CUDA_LIB_DIR/cublasLt.lib

# =========================
# Pre-link hook: build CUDA object with NVCC
# =========================
NVCC_EXE     = $$CUDA_PATH/bin/nvcc.exe
CUDA_SRC     = $$PWD/src/mri_engine.cu
CUDA_OBJ_REL = obj/mri_engine_cuda.obj

message([DBG] NVCC_EXE = $$NVCC_EXE)
message([DBG] OBJECTS_DIR = $$OBJECTS_DIR)
message([DBG] Forcing link to include CUDA_OBJ = $$CUDA_OBJ_REL)

# Ensure the linker expects the object and Clean removes it
OBJECTS        += $$CUDA_OBJ_REL
PRE_TARGETDEPS += $$CUDA_OBJ_REL
QMAKE_CLEAN    += $$CUDA_OBJ_REL

# Compile .cu right before link (single -Xcompiler token)
QMAKE_PRE_LINK = $$NVCC_EXE -c $$CUDA_SRC -o $$CUDA_OBJ_REL \
                 --use-local-env -std=c++17 -g -G -lineinfo \
                 -Xcompiler=/Z7,/MDd,/EHsc,/W3 \
                 -DWIN32 -D_WINDOWS -DMRI_ENGINE_BUILD \
                 -arch=sm_$$CUDA_ARCH \
                 -I$$PWD/include -I$$CUDA_INC_DIR

# =========================
# MSVC flags for C++ sources
# =========================
QMAKE_CXXFLAGS_DEBUG += /Z7
QMAKE_LFLAGS_DEBUG   += /DEBUG
DEFINES += MRI_ENGINE_BUILD

# =========================
# Output dir & optional post-link copies
# =========================
DESTDIR = $$PWD/bin
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] Built $$TARGET to %CD%)

