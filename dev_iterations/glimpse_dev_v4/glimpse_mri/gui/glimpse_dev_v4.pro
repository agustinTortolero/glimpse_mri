# =========================
# App .pro (DEBUG ONLY)
# =========================
QT += core gui widgets
CONFIG += c++17
TEMPLATE = app

OBJECTS_DIR = $$OUT_PWD/obj
MOC_DIR     = $$OUT_PWD/moc
RCC_DIR     = $$OUT_PWD/rcc
UI_DIR      = $$OUT_PWD/ui

# -------- sources --------
SOURCES += \
    model/dicom_dll.cpp \
    model/io.cpp \
    src/main.cpp \
    controller/app_controller.cpp \
    view/mainwindow.cpp

HEADERS += \
    model/dicom_dll.hpp \
    model/io.hpp \
    controller/app_controller.hpp \
    src/logger.hpp \
    view/mainwindow.hpp \
    src/image_utils.hpp


# make headers in model/ visible to all sources

# -------- vcpkg --------
VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
VCPKG_INC  = $$VCPKG_ROOT/include
VCPKG_LIBD = $$VCPKG_ROOT/debug/lib
VCPKG_BIND = $$VCPKG_ROOT/debug/bin

INCLUDEPATH += $$VCPKG_INC
DEFINES += _CRT_SECURE_NO_WARNINGS
DEFINES += HAVE_ISMRMRD

# Enable DCMTK JPEG/JLS by default (you can flip to 0 if a lib is missing)
DEFINES += IO_ENABLE_DCMJPEG=1 IO_ENABLE_DCMJPLS=1 IO_ENABLE_DCMJ2K=0

QMAKE_LFLAGS_DEBUG += /VERBOSE:LIB
QMAKE_CXXFLAGS += /bigobj

# ---- OpenCV (debug) ----
OPENCV_INC = C:/opencv/opencv/build/include
OPENCV_LIB = C:/opencv/opencv/build/x64/vc16/lib
INCLUDEPATH += $$OPENCV_INC
win32:CONFIG(debug, debug|release):LIBS += -L$$OPENCV_LIB -lopencv_world490d -ladvapi32

# =========================
# MRI ENGINE DLL (Debug)
# =========================

# (A) Choose the header folder (prefer the one under Glimpse/dev; fall back to EXPERIMENTS)
ENGINE_INC_PRIMARY  = C:/AgustinTortolero_repos/MRI/Glimpse/dev_iterations/mri_engine/mri_engine_v_1_1/include
ENGINE_INC_FALLBACK = C:/AgustinTortolero_repos/MRI/MRI_EXPERIMENTS/engine/engine_1/mri_engine_v_1_1/include

MRI_DLL_INC = $$ENGINE_INC_PRIMARY
win32:!exists($$MRI_DLL_INC) {
    MRI_DLL_INC = $$ENGINE_INC_FALLBACK
    message("[WARN][App] Primary engine include not found; using fallback: $$MRI_DLL_INC")
}

# Put engine include FIRST so it wins shadowing
INCLUDEPATH = $$MRI_DLL_INC $$INCLUDEPATH
DEPENDPATH  = $$MRI_DLL_INC $$DEPENDPATH
message("[DBG][App] Using engine include: $$MRI_DLL_INC")

# (B) Library/DLL folder — you said this is the actual Debug path
MRI_DLL_DIR = C:/AgustinTortolero_repos/MRI/MRI_EXPERIMENTS/engine/engine_1/build/Debug
message("[DBG][App] MRI_DLL_DIR = $$MRI_DLL_DIR")

# Link against the import lib that sits next to the DLL
win32:exists($$MRI_DLL_DIR) {
    LIBS += -L"$$MRI_DLL_DIR" -lmri_engine_v_1_1
    message("[DBG][App] Linking with mri_engine_v_1_1.lib from $$MRI_DLL_DIR")
} else {
    message("[ERR][App] MRI_DLL_DIR NOT FOUND: $$MRI_DLL_DIR (not adding -L)")
}

# (C) Post-link: copy the DLL next to the app (debug)
DEST_DLL_DIR = $$OUT_PWD/debug
QMAKE_POST_LINK += $$quote(cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR")
win32:exists($$MRI_DLL_DIR) {
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$MRI_DLL_DIR\\mri_engine_v_1_1.dll" "$$DEST_DLL_DIR\\")
}
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] Engine + deps copied to "$$DEST_DLL_DIR")

# =========================
# vcpkg libs (Debug)
# =========================
LIBS += -L$$VCPKG_LIBD

# ISMRMRD + deps
LIBS += -lismrmrd -lpugixml
LIBS += -lhdf5_D -lhdf5_hl_D -lhdf5_cpp_D -lhdf5_hl_cpp_D
LIBS += -lzlibd -laec

# DCMTK core
message("[DBG] DCMTK: linking dcmimgle, dcmimage, dcmdata, oflog, ofstd")
LIBS += -ldcmimgle -ldcmimage -ldcmdata -loflog -lofstd

# DCMTK decoders (JPEG/JLS). Comment out blocks if your vcpkg DCMTK lacks them.
# JPEG baseline/extended/lossless
LIBS += -ldcmjpeg -lijg8 -lijg12 -lijg16
# JPEG-LS
LIBS += -ldcmjpls -ldcmtkcharls
# JPEG 2000 (enable if you flip IO_ENABLE_DCMJ2K=1 and have the libs)
# LIBS += -ldcmj2k -lopenjp2

LIBS += -lshell32 -lole32

# =========================
# RUNTIME DLL COPY (Debug)
# =========================
DEST_DLL_DIR = $$OUT_PWD/debug

QMAKE_POST_LINK =
QMAKE_POST_LINK += $$quote(cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR")
QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] Copying runtime DLLs to "$$DEST_DLL_DIR")

# Engine DLL
win32:exists($$MRI_DLL_DIR) {
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$MRI_DLL_DIR\\mri_engine_v_1_1.dll" "$$DEST_DLL_DIR\\")
}

# vcpkg runtime DLLs your app needs
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\ismrmrd*.dll" "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\pugixml*.dll" "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\hdf5*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\zlib*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\*aec*.dll"   "$$DEST_DLL_DIR\\")

# DCMTK core + decoders
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\dcm*.dll"    "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\of*.dll"     "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\ijg*.dll"    "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\*charls*.dll" "$$DEST_DLL_DIR\\")
# Uncomment if using JPEG2000
# QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BIND\\openjp2*.dll" "$$DEST_DLL_DIR\\")

# OpenCV runtime (debug)
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/opencv/opencv/build/x64/vc16/bin/opencv_world*d.dll" "$$DEST_DLL_DIR\\")

# CUDA runtime DLLs (required by your engine DLL)
CUDA_PATH = C:/PROGRA~1/NVIDIA~2/CUDA/v12.4
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cudart64*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cufft64*.dll"    "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cublas64*.dll"   "$$DEST_DLL_DIR\\")
QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cublasLt64*.dll" "$$DEST_DLL_DIR\\")

QMAKE_POST_LINK += $$quote(cmd /c echo [DBG] DLLs now in "$$DEST_DLL_DIR":)
QMAKE_POST_LINK += $$quote(cmd /c dir /b "$$DEST_DLL_DIR\\*.dll")
