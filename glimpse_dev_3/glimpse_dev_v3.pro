# =========================
# glimpse_dev_v1.pro (DEBUG ONLY) — DLL-only MRI engine
# =========================

QT += core gui widgets
CONFIG += c++17
TEMPLATE = app

# Predictable build dirs
OBJECTS_DIR = $$OUT_PWD/obj
MOC_DIR     = $$OUT_PWD/moc
RCC_DIR     = $$OUT_PWD/rcc
UI_DIR      = $$OUT_PWD/ui

# -------- sources --------
SOURCES += \
    model/io.cpp \
    model/io_ismrmrd.cpp \
    src/main.cpp \
    controller/app_controller.cpp \
    model/io_fastmri.cpp \
    view/mainwindow.cpp

HEADERS += \
    model/io.hpp \
    model/io_ismrmrd.hpp \
    src/common.hpp \
    model/io_fastmri.hpp \
    controller/app_controller.hpp \
    view/mainwindow.hpp \
    src/image_utils.hpp
# NOTE: do not list model/mri_engine.hpp here; use the DLL's mri_engine_api.h instead.

# -------- vcpkg (forward slashes) --------
VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
VCPKG_INC  = $$VCPKG_ROOT/include
VCPKG_LIBD = $$VCPKG_ROOT/debug/lib
VCPKG_BIND = $$VCPKG_ROOT/debug/bin

INCLUDEPATH += $$VCPKG_INC
DEFINES += _CRT_SECURE_NO_WARNINGS
DEFINES += HAVE_ISMRMRD

message("[DBG] VCPKG_INC  = $$VCPKG_INC")
message("[DBG] VCPKG_LIBD = $$VCPKG_LIBD")
message("[DBG] VCPKG_BIND = $$VCPKG_BIND")
message("[DBG] Build = DEBUG")

QMAKE_LFLAGS_DEBUG += /VERBOSE:LIB
QMAKE_CXXFLAGS += /bigobj

# ---------------- OpenMP ----------------
win32:msvc {
    QMAKE_CXXFLAGS += /openmp:llvm
    DEFINES += OMP_ENABLED=1
    message("[DBG] OpenMP: MSVC /openmp:llvm (compiler only)")
}
win32:!msvc {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS   += -fopenmp
    DEFINES += OMP_ENABLED=1
    message("[DBG] OpenMP: MinGW/Clang -fopenmp")
}
unix:!win32 {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS   += -fopenmp
    DEFINES += OMP_ENABLED=1
    message("[DBG] OpenMP: unix -fopenmp")
}

# ---- OpenCV ----
OPENCV_INC = C:/opencv/opencv/build/include
OPENCV_LIB = C:/opencv/opencv/build/x64/vc16/lib
INCLUDEPATH += $$OPENCV_INC
win32:CONFIG(debug, debug|release):LIBS += -L$$OPENCV_LIB -lopencv_world490d -ladvapi32

# =========================
# MRI ENGINE (DLL-only) —— NO CUDA build here
# =========================
MRI_ENGINE_ROOT = C:/AgustinTortolero_repos/MRI/Glimpse/dev_iterations/mri__engine_lib_dev_1
MRI_DLL_INC    = $$MRI_ENGINE_ROOT/include
MRI_DLL_BIN    = $$MRI_ENGINE_ROOT/bin

message("[DBG][App] MRI_ENGINE_ROOT = $$MRI_ENGINE_ROOT")
message("[DBG][App] MRI_DLL_INC    = $$MRI_DLL_INC")
message("[DBG][App] MRI_DLL_BIN    = $$MRI_DLL_BIN")

# Use the C ABI header from the DLL
INCLUDEPATH += $$MRI_DLL_INC

# Compile out any in-process fallback paths
DEFINES += MRI_DLL_ONLY

# =========================
# vcpkg libs (Debug)
# =========================
LIBS += -L$$VCPKG_LIBD
# ISMRMRD + deps
LIBS += -lismrmrd -lpugixml
LIBS += -lhdf5_D -lhdf5_hl_D -lhdf5_cpp_D -lhdf5_hl_cpp_D
LIBS += -lzlibd -laec

# DCMTK (debug)
message("[DBG] DCMTK: linking dcmimgle, dcmimage, dcmdata, oflog, ofstd")
LIBS += -ldcmimgle -ldcmimage -ldcmdata -loflog -lofstd
# If your DICOMs are JPEG-compressed, you may also need:
# LIBS += -ldcmjpeg -lcharls -ljpeg -lpng -ltiff -lz

# =========================
# RUNTIME DLL COPY (Debug)
# =========================
DEST_DLL_DIR = $$OUT_PWD/debug

QMAKE_POST_LINK =
QMAKE_POST_LINK += $$quote(cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR")
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] Copying runtime DLLs to "$$DEST_DLL_DIR")

# 1) Copy the MRI engine DLL next to the app
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$MRI_DLL_BIN\\mri__engine_lib.dll" "$$DEST_DLL_DIR\\")

# 2) vcpkg DLLs needed by ISMRMRD/HDF5/etc
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\ismrmrd*.dll" "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\pugixml*.dll" "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\hdf5*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\zlib*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\*aec*.dll"   "$$DEST_DLL_DIR\\")

# 3) DCMTK (debug)
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\dcm*.dll" "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\of*.dll"  "$$DEST_DLL_DIR\\")

# 4) OpenCV (debug)
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/opencv/opencv/build/x64/vc16/bin/opencv_world*d.dll" "$$DEST_DLL_DIR\\")

# 5) CUDA runtime DLLs (needed by the DLL; app itself doesn't link CUDA)
CUDA_PATH = C:/PROGRA~1/NVIDIA~2/CUDA/v12.4
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cudart64*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cufft64*.dll"    "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cublas64*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cublasLt64*.dll" "$$DEST_DLL_DIR\\")

# final listing
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] DLLs now in "$$DEST_DLL_DIR":)
QMAKE_POST_LINK += $$quote(cmd /c dir /b "$$DEST_DLL_DIR\\*.dll")

# =========================
# IMPORTANT: No CUDA compilation in this app
# =========================
# Ensure these lines DO NOT exist anywhere:
#   CUDA_SOURCES += model/mri_engine.cu
#   QMAKE_EXTRA_COMPILERS += cuda
#   NVCC = ...
# And do NOT link CUDA libs here; the DLL uses them, not the app.
