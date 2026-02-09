# ============================================================
# Glimpse MRI — Qt 6 — RELEASE ONLY — Windows + Linux (Jetson)
# ============================================================
QT += core gui widgets
TEMPLATE = app
CONFIG  += c++17 release
CONFIG  -= debug

# ============================================================
# Version / Build fingerprint (auto embedded into the binary)
# ============================================================

# Single source of truth for GUI version
VERSION = 1.1.0

# Build type (even if you force release, it will show Release)
CONFIG(debug, debug|release) {
    BUILD_TYPE = Debug
} else {
    BUILD_TYPE = Release
}

CONFIG(debug, debug|release) {
    DEFINES += SHOW_QMAKE_PATH=1
}



# Qt version (qmake built-in)
QT_BUILD_VERSION = $$QT_VERSION

# qmake executable path (qmake built-in)
QMAKE_PATH = $$QMAKE_QMAKE

# qmake version (QMAKE_VERSION is sometimes empty on Windows, so query qmake)
QMAKE_BUILD_VERSION = $$QMAKE_VERSION
win32:isEmpty(QMAKE_BUILD_VERSION) {
    QMAKE_BUILD_VERSION = $$system($$QMAKE_QMAKE -query QMAKE_VERSION)
}
unix:!win32:isEmpty(QMAKE_BUILD_VERSION) {
    QMAKE_BUILD_VERSION = $$system($$QMAKE_QMAKE -query QMAKE_VERSION 2>/dev/null)
}

# Git + Timestamp (best-effort; no build break if missing)
GIT_BIN = git
GIT_SHA = Unknown
GIT_DESCRIBE = Unknown
BUILD_TIMESTAMP = Unknown

win32 {
    # Use a Windows-native path for -C (backslashes)
    WIN_PWD = $$system_path($$PWD)

    # Prefer well-known Git for Windows locations (works even if PATH is missing)
    exists("C:/Program Files/Git/cmd/git.exe") {
        GIT_BIN = "C:/Program Files/Git/cmd/git.exe"
    } else:exists("C:/Program Files/Git/bin/git.exe") {
        GIT_BIN = "C:/Program Files/Git/bin/git.exe"
    } else:exists("C:/Program Files (x86)/Git/cmd/git.exe") {
        GIT_BIN = "C:/Program Files (x86)/Git/cmd/git.exe"
    } else:exists("C:/Program Files (x86)/Git/bin/git.exe") {
        GIT_BIN = "C:/Program Files (x86)/Git/bin/git.exe"
    } else {
        # Fallback (only works if git is on PATH)
        GIT_BIN = git
    }

    # Optional: check repo status (helps debugging)
    GIT_IS_REPO = $$system("\"$$GIT_BIN\" -C \"$$WIN_PWD\" rev-parse --is-inside-work-tree 2>NUL")
    GIT_IS_REPO = $$replace(GIT_IS_REPO, \\s, )

    contains(GIT_IS_REPO, true) {
        # Commit hash (12 chars) + describe
        GIT_SHA      = $$system("\"$$GIT_BIN\" -C \"$$WIN_PWD\" rev-parse --short=12 HEAD 2>NUL")
        GIT_DESCRIBE = $$system("\"$$GIT_BIN\" -C \"$$WIN_PWD\" describe --tags --always --dirty --abbrev=12 2>NUL")
    }

    # Timestamp (no newline)
    BUILD_TIMESTAMP = $$system(powershell -NoProfile -Command "[Console]::Write((Get-Date).ToString('yyyy-MM-ddTHH:mm:ss'))")

    # Debug visibility during qmake
    message([ver] WIN_PWD=$$WIN_PWD)
    message([ver] GIT_BIN=$$GIT_BIN)
    message([ver] GIT_IS_REPO=$$GIT_IS_REPO)

QT_BIN = $$[QT_INSTALL_BINS]
WINDEPLOYQT = $$QT_BIN/windeployqt.exe

# After building and staging your own DLLs, deploy Qt runtime next to the exe
exists($$WINDEPLOYQT) {
    message([win][qt] windeployqt=$$WINDEPLOYQT)

    # Adjust exe name/path if yours differs
    TARGET_EXE = $$OUT_PWD/release/glimpseMRI.exe

    QMAKE_POST_LINK += $$quote("\"$$WINDEPLOYQT\" --release --no-translations --compiler-runtime \"$$TARGET_EXE\"")
    QMAKE_POST_LINK += $$quote(cmd /c echo [qt] windeployqt finished)
} else {
    message([win][qt][WARN] windeployqt not found at $$WINDEPLOYQT)
}


}


unix:!win32 {
    GIT_SHA      = $$system(git -C $$PWD rev-parse --short=12 HEAD 2>/dev/null)
    GIT_DESCRIBE = $$system(git -C $$PWD describe --tags --always --dirty 2>/dev/null)
    BUILD_TIMESTAMP = $$system(date "+%Y-%m-%dT%H:%M:%S")
}

# Robust cleanup: remove ALL whitespace (CR/LF/tabs/spaces)
GIT_SHA            = $$replace(GIT_SHA, \\s, )
GIT_DESCRIBE       = $$replace(GIT_DESCRIBE, \\s, )
BUILD_TIMESTAMP    = $$replace(BUILD_TIMESTAMP, \\s, )
QMAKE_BUILD_VERSION= $$replace(QMAKE_BUILD_VERSION, \\s, )


# Keep spaces in paths; remove only CR/LF/TAB
GIT_BIN   = $$replace(GIT_BIN, \\r, )
GIT_BIN   = $$replace(GIT_BIN, \\n, )
GIT_BIN   = $$replace(GIT_BIN, \\t, )
QMAKE_PATH= $$replace(QMAKE_PATH, \\r, )
QMAKE_PATH= $$replace(QMAKE_PATH, \\n, )
QMAKE_PATH= $$replace(QMAKE_PATH, \\t, )
# Extra safety: remove quotes if any
GIT_BIN            = $$replace(GIT_BIN, \", )
GIT_SHA            = $$replace(GIT_SHA, \", )
GIT_DESCRIBE       = $$replace(GIT_DESCRIBE, \", )
BUILD_TIMESTAMP    = $$replace(BUILD_TIMESTAMP, \", )
QMAKE_BUILD_VERSION= $$replace(QMAKE_BUILD_VERSION, \", )
QMAKE_PATH         = $$replace(QMAKE_PATH, \", )

isEmpty(GIT_SHA)            { GIT_SHA = Unknown }
isEmpty(GIT_DESCRIBE)       { GIT_DESCRIBE = Unknown }
isEmpty(BUILD_TIMESTAMP)    { BUILD_TIMESTAMP = Unknown }
isEmpty(QMAKE_BUILD_VERSION){ QMAKE_BUILD_VERSION = Unknown }
isEmpty(QMAKE_PATH)         { QMAKE_PATH = Unknown }

# Export as C/C++ macros (string literals)
DEFINES += GUI_VERSION_STR=\\\"$$VERSION\\\"
DEFINES += BUILD_TYPE_STR=\\\"$$BUILD_TYPE\\\"
DEFINES += GIT_SHA_STR=\\\"$$GIT_SHA\\\"
DEFINES += GIT_DESCRIBE_STR=\\\"$$GIT_DESCRIBE\\\"
DEFINES += BUILD_TIMESTAMP_STR=\\\"$$BUILD_TIMESTAMP\\\"
DEFINES += QT_BUILD_VERSION_STR=\\\"$$QT_BUILD_VERSION\\\"
DEFINES += QMAKE_BUILD_VERSION_STR=\\\"$$QMAKE_BUILD_VERSION\\\"
DEFINES += QMAKE_PATH_STR=\\\"$$QMAKE_QMAKE\\\"

# Visible during qmake step
message([ver] GUI_VERSION=$$VERSION)
message([ver] BUILD_TYPE=$$BUILD_TYPE)
message([ver] QT_BUILD_VERSION=$$QT_BUILD_VERSION)
message([ver] QMAKE_BUILD_VERSION=$$QMAKE_BUILD_VERSION)
message([ver] GIT_DESCRIBE=$$GIT_DESCRIBE)
message([ver] GIT_SHA=$$GIT_SHA)
message([ver] BUILD_TIMESTAMP=$$BUILD_TIMESTAMP)



# ---------- Common project layout ----------
OBJECTS_DIR = $$OUT_PWD/obj
MOC_DIR     = $$OUT_PWD/moc
RCC_DIR     = $$OUT_PWD/rcc
UI_DIR      = $$OUT_PWD/ui

INCLUDEPATH += $$PWD $$PWD/model $$PWD/controller $$PWD/view $$PWD/src
DEPENDPATH  += $$PWD $$PWD/model $$PWD/controller $$PWD/view $$PWD/src
RESOURCES   += assets/assets.qrc

message([os] qmake host/platform selects proper block below)

# ---------- Sources / Headers ----------
SOURCES += \
    model/io.cpp \
    src/main.cpp \
    controller/app_controller.cpp \
    view/mainwindow.cpp \
    view/progress_splash.cpp

HEADERS += \
    model/engine_api.h \
    model/io.hpp \
    controller/app_controller.hpp \
    src/build_info.hpp \
    src/build_info.hpp \
    src/logger.hpp \
    view/mainwindow.hpp \
    src/image_utils.hpp \
    view/progress_splash.hpp

# ============================================================
# WINDOWS (MSVC) — RELEASE
# ============================================================
win32 {
# --- Windows resource: icon + version info (hardcoded path in RC) ---
RC_FILE = assets/app_win.rc
message([win][rc] using RC_FILE assets/app_win.rc)




    message([win] ===== Windows/MSVC RELEASE configuration =====)

    # ---- vcpkg (release) ----
    VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
    VCPKG_INC  = $$VCPKG_ROOT/include
    VCPKG_LIBR = $$VCPKG_ROOT/lib
    VCPKG_BINR = $$VCPKG_ROOT/bin

    INCLUDEPATH += $$VCPKG_INC
    message([win][vcpkg] INC=$$VCPKG_INC  LIBR=$$VCPKG_LIBR  BINR=$$VCPKG_BINR)

    DEFINES += _CRT_SECURE_NO_WARNINGS
    DEFINES += HAVE_ISMRMRD
    DEFINES += IO_ENABLE_DCMJPEG=1 IO_ENABLE_DCMJPLS=1 IO_ENABLE_DCMJ2K=0

    QMAKE_CXXFLAGS += /bigobj /O2 /DNDEBUG
    QMAKE_LFLAGS   += /INCREMENTAL:NO

    # ---- OpenCV (exact paths; release) ----
    OPENCV_INC = C:/opencv/opencv/build/include
    OPENCV_LIB = C:/opencv/opencv/build/x64/vc16/lib
    INCLUDEPATH += $$OPENCV_INC
    message([win][opencv] INC=$$OPENCV_INC  LIB=$$OPENCV_LIB)

    LIBS += -L$$OPENCV_LIB -lopencv_world490 -ladvapi32
    message([win][opencv] linking opencv_world490 + advapi32)

    # ---- MRI Engine & DICOM (GUI-local release libs) ----

    ENGINE_LIB_RELEASE = $$PWD/../engine/build/Release/mri_engine.lib
    ENGINE_DLL_RELEASE = $$PWD/../engine/build/Release/mri_engine.dll
    LIBS += $$ENGINE_LIB_RELEASE

    DICOM_LIB_RELEASE  = $$PWD/release/dicom_io_lib.lib
    DICOM_DLL_RELEASE  = $$PWD/release/dicom_io_lib.dll

    QMAKE_LIBDIR += $$PWD/release
    message([win][libdir] + $$PWD/release)

    exists($$ENGINE_LIB_RELEASE) {
       LIBS += "$$ENGINE_LIB_RELEASE"
       message([win][engine][release] +link $$ENGINE_LIB_RELEASE)
    } else {
      message([win][engine][release][ERR] missing import lib: $$ENGINE_LIB_RELEASE)
    }


    exists($$DICOM_LIB_RELEASE) {
        LIBS += "$$DICOM_LIB_RELEASE"
        message([win][dicom ][release] +link $$DICOM_LIB_RELEASE)
    } else {
        message([win][dicom ][release][ERR] missing import lib: $$DICOM_LIB_RELEASE)
    }

    # ---- vcpkg release libs ----
    LIBS += -L$$VCPKG_LIBR
    LIBS += -lismrmrd -lpugixml
    LIBS += -lhdf5 -lhdf5_hl -lhdf5_cpp -lhdf5_hl_cpp
    LIBS += -lzlib -laec
    message([win][DCMTK] link dcmimgle dcmimage dcmdata oflog ofstd + decoders)
    LIBS += -ldcmimgle -ldcmimage -ldcmdata -loflog -lofstd
    LIBS += -ldcmjpeg -lijg8 -lijg12 -lijg16
    LIBS += -ldcmjpls -ldcmtkcharls
    # LIBS += -ldcmj2k -lopenjp2    # if JPEG2000
    LIBS += -lshell32 -lole32

    # ---- Post-link staging (Windows only) ----
    DEST_DLL_DIR = $$OUT_PWD/release
    message([win][stage] DEST_DLL_DIR = $$DEST_DLL_DIR)

    QMAKE_POST_LINK += $$quote(cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR")
    QMAKE_POST_LINK += $$quote(cmd /c echo [stage] Copying runtime DLLs -> "$$DEST_DLL_DIR")

    exists($$ENGINE_DLL_RELEASE) {
     message([win][engine][release] stage $$ENGINE_DLL_RELEASE -> $$DEST_DLL_DIR)
     # [DBG] Keep original file name from ENGINE_DLL_RELEASE (mri_engine.dll)
        QMAKE_POST_LINK += $$quote(cmd /c copy /Y "$$ENGINE_DLL_RELEASE" "$$DEST_DLL_DIR\\" >nul)
    } else {
        message([win][engine][release][WARN] missing DLL: $$ENGINE_DLL_RELEASE)
    }
    exists($$DICOM_DLL_RELEASE) {
        message([win][dicom ][release] stage $$DICOM_DLL_RELEASE -> $$DEST_DLL_DIR)
        QMAKE_POST_LINK += $$quote(cmd /c copy /Y "$$DICOM_DLL_RELEASE" "$$DEST_DLL_DIR\\dicom_io_lib.dll" >nul)
    } else {
        message([win][dicom ][release][WARN] missing DLL: $$DICOM_DLL_RELEASE)
    }

    # OpenCV runtime (release)
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "C:/opencv/opencv/build/x64/vc16/bin/opencv_world490.dll" "$$DEST_DLL_DIR\\")

    # vcpkg runtimes (release)
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\ismrmrd*.dll" "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\pugixml*.dll" "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\hdf5*.dll"   "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\zlib*.dll"   "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\*aec*.dll"   "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\dcm*.dll"    "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\of*.dll"     "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /Y /I /Q "$$VCPKG_BINR\\ijg*.dll"    "$$DEST_DLL_DIR\\")

    # CUDA runtime DLLs (only if your engine needs them)
    CUDA_PATH = C:/PROGRA~1/NVIDIA~2/CUDA/v12.4
    QMAKE_POST_LINK += $$quote(cmd /c if exist "$$CUDA_PATH\\bin\\cudart64*.dll"   xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cudart64*.dll"   "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c if exist "$$CUDA_PATH\\bin\\cufft64*.dll"    xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cufft64*.dll"    "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c if exist "$$CUDA_PATH\\bin\\cublas64*.dll"   xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cublas64*.dll"   "$$DEST_DLL_DIR\\")
    QMAKE_POST_LINK += $$quote(cmd /c if exist "$$CUDA_PATH\\bin\\cublasLt64*.dll" xcopy /Y /I /Q "$$CUDA_PATH\\bin\\cublasLt64*.dll" "$$DEST_DLL_DIR\\")

    QMAKE_POST_LINK += $$quote(cmd /c echo [stage] DLLs now in "$$DEST_DLL_DIR":)
    QMAKE_POST_LINK += $$quote(cmd /c dir /b "$$DEST_DLL_DIR\\*.dll")
}

# ============================================================
# LINUX / Jetson (aarch64 Ubuntu) — RELEASE
# ============================================================
unix:!win32 {
    message([lin] ===== Linux/Jetson RELEASE configuration =====)

    DEFINES += GLIMPSE_JETSON
    DEFINES += HAVE_ISMRMRD
    DEFINES += IO_ENABLE_DCMJPEG=1 IO_ENABLE_DCMJPLS=1 IO_ENABLE_DCMJ2K=0

    # Toolchain defaults already set - just reinforce optimization & NDEBUG
    QMAKE_CXXFLAGS += -O3 -DNDEBUG -fPIC
    QMAKE_LFLAGS   += -Wl,--as-needed

    # ---- Prefer pkg-config wherever possible ----
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv4 ismrmrd pugixml
    # HDF5 packages commonly present on Ubuntu/Jetson
    PKGCONFIG += hdf5 hdf5_hl hdf5_cpp hdf5_hl_cpp

    message([lin][pkg] using pkg-config: $$PKGCONFIG)

    # ---- DCMTK: link components explicitly (common split on Ubuntu) ----
    # If your DCMTK was built differently, adjust this list.
    LIBS += -ldcmimgle -ldcmimage -ldcmdata -loflog -lofstd
    LIBS += -ldcmjpeg -ldcmjpls
    # Enable if you have JPEG2000:
    # LIBS += -ldcmj2k -lopenjp2

    # ---- MRI Engine & DICOM (GUI-local release .so) ----
    # Expected filenames (adjust if your .so names differ)
    # [DBG] Engine now built as versionless libmri_engine.so
    ENGINE_SO_PATH = $$PWD/release/libmri_engine.so
    DICOM_SO_PATH  = $$PWD/release/libdicom_io_lib.so

    QMAKE_LIBDIR += $$PWD/release
    message([lin][libdir] + $$PWD/release)

# Link using -l... only if the .so exists locally (keep build portable)
    exists($$ENGINE_SO_PATH) {
        LIBS += -L$$PWD/release -lmri_engine
        message([lin][engine] +link -lmri_engine (found $$ENGINE_SO_PATH))
    } else {
        message([lin][engine][WARN] engine .so not found at $$ENGINE_SO_PATH (skipping -l))
    }

    exists($$DICOM_SO_PATH) {
        LIBS += -L$$PWD/release -ldicom_io_lib
        message([lin][dicom ] +link -ldicom_io_lib (found $$DICOM_SO_PATH))
    } else {
        message([lin][dicom ][WARN] dicom .so not found at $$DICOM_SO_PATH (skipping -l))
    }

    # ---- RPATH so the app finds local engine/dicom .so at runtime ----
    # Also include CUDA default location on Jetson.
    CUDA_PATH = /usr/local/cuda
    QMAKE_RPATHDIR += $$OUT_PWD/release $$PWD/release $$CUDA_PATH/lib64
    QMAKE_LFLAGS   += -Wl,-rpath,$$OUT_PWD/release -Wl,-rpath,$$PWD/release -Wl,-rpath,$$CUDA_PATH/lib64

    message([lin][rpath] OUT=$$OUT_PWD/release  PWD=$$PWD/release  CUDA=$$CUDA_PATH/lib64)

    # ---- (Optional) post-link listing for visibility ----
    DEST_SO_DIR = $$OUT_PWD/release
    QMAKE_POST_LINK += echo "[stage] Linux release output in $$DEST_SO_DIR"
    QMAKE_POST_LINK += && ls -1 "$$DEST_SO_DIR" || true
}

DISTFILES +=
