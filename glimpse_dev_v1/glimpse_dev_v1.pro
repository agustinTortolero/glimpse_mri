# =========================
# glimpse_dev_v1.pro (DEBUG ONLY)
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
    src/main.cpp \
    controller/app_controller.cpp \
    model/io_fastmri.cpp \
    view/mainwindow.cpp

HEADERS += \
    src/common.hpp \
    model/mri_engine.hpp \
    model/io_fastmri.hpp \
    controller/app_controller.hpp \
    view/mainwindow.hpp \
    src/image_utils.hpp

# -------- vcpkg (forward slashes) --------
VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
VCPKG_INC  = $$VCPKG_ROOT/include
VCPKG_LIBD = $$VCPKG_ROOT/debug/lib

INCLUDEPATH += $$VCPKG_INC
DEFINES += _CRT_SECURE_NO_WARNINGS

message([DBG] VCPKG_INC  = $$VCPKG_INC)
message([DBG] VCPKG_LIBD = $$VCPKG_LIBD)
message([DBG] Build = DEBUG)

QMAKE_LFLAGS_DEBUG += /VERBOSE:LIB
QMAKE_CXXFLAGS += /bigobj

# ---- OpenCV ----
OPENCV_INC = C:/opencv/opencv/build/include
OPENCV_LIB = C:/opencv/opencv/build/x64/vc16/lib
INCLUDEPATH += $$OPENCV_INC
# Use Debug lib (adjust version if needed)
win32:CONFIG(debug, debug|release):LIBS += -L$$OPENCV_LIB -lopencv_world490d -ladvapi32

# ---- CUDA (8.3 short path confirmed) ----
CUDA_PATH = C:/PROGRA~1/NVIDIA~2/CUDA/v12.4
CUDA_INC  = $$CUDA_PATH/include
CUDA_LIB  = $$CUDA_PATH/lib/x64
NVCC      = $$CUDA_PATH/bin/nvcc.exe

# Do NOT add CUDA_INC to global INCLUDEPATH (host .cpps don't need it)
win32:LIBS += -L$$CUDA_LIB -lcudart -lcuda -lcufft -lcublas

# ---- HDF5 + DCMTK (DEBUG from vcpkg) ----
LIBS += -L$$VCPKG_LIBD
LIBS += -lhdf5_D -lhdf5_hl_D -lhdf5_cpp_D -lhdf5_hl_cpp_D
LIBS += -lzlibd -laec
LIBS += -ldcmdata -lofstd -loflog

# ---- CUDA (.cu) via nvcc ----
CUDA_SOURCES += model/mri_engine.cu

cuda.input  = CUDA_SOURCES
cuda.output = $$OUT_PWD/obj/${QMAKE_FILE_BASE}_cuda.obj
cuda.dependency_type = TYPE_C
cuda.variable_out = OBJECTS
cuda.commands = "$$NVCC" -c "${QMAKE_FILE_NAME}" -o "${QMAKE_FILE_OUT}" -std=c++17 \
    -Xcompiler="/EHsc" -Xcompiler="/MDd" -Xcompiler="/nologo" \
    -I"$$CUDA_INC" -I"$$VCPKG_INC" -I"$$PWD/src" -I"$$PWD" -I"$$OPENCV_INC"
QMAKE_EXTRA_COMPILERS += cuda

# ---- Post-link: copy runtime DLLs ----
DEST_DLL_DIR = $$OUT_PWD/debug

# reset, then add one command per line (more reliable than long && chains)
QMAKE_POST_LINK =
QMAKE_POST_LINK += $$quote(cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR")
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] Copying runtime DLLs to "$$DEST_DLL_DIR")

# HDF5 + dependencies (debug)
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/src/vcpkg/installed/x64-windows/debug/bin/hdf5*.dll" "$$DEST_DLL_DIR/")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/src/vcpkg/installed/x64-windows/debug/bin/zlib*.dll" "$$DEST_DLL_DIR/")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/src/vcpkg/installed/x64-windows/debug/bin/*aec*.dll" "$$DEST_DLL_DIR/")

# DCMTK (debug)
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/src/vcpkg/installed/x64-windows/debug/bin/dcm*.dll" "$$DEST_DLL_DIR/")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/src/vcpkg/installed/x64-windows/debug/bin/of*.dll"  "$$DEST_DLL_DIR/")

# OpenCV (debug) — wildcard to survive minor version bumps
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/opencv/opencv/build/x64/vc16/bin/opencv_world*d.dll" "$$DEST_DLL_DIR/")

# CUDA
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH/bin/cudart64*.dll"   "$$DEST_DLL_DIR/")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH/bin/cufft64*.dll"    "$$DEST_DLL_DIR/")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH/bin/cublas64*.dll"   "$$DEST_DLL_DIR/")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH/bin/cublasLt64*.dll" "$$DEST_DLL_DIR/")

# final listing
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] DLLs now in "$$DEST_DLL_DIR":)
QMAKE_POST_LINK += $$quote(cmd /c dir /b "$$DEST_DLL_DIR\\*.dll")
