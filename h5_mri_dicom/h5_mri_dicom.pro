QT += core
CONFIG += console c++17
CONFIG -= app_bundle
TEMPLATE = app

SOURCES += \
    src/main.cpp \
    src/read_hdf5.cpp \
    src/fastmri_reader.cpp \
    src/ismrmrd_reader.cpp \
    src/utils.cpp

HEADERS += \
    src/read_hdf5.h \
    src/fastmri_reader.h \
    src/ismrmrd_reader.h \
    src/utils.h

# ---- vcpkg roots (forward slashes) ----
VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
VCPKG_INC  = $$VCPKG_ROOT/include
VCPKG_LIBR = $$VCPKG_ROOT/lib
VCPKG_LIBD = $$VCPKG_ROOT/debug/lib

INCLUDEPATH += $$VCPKG_INC
DEFINES += _CRT_SECURE_NO_WARNINGS

message([DBG] VCPKG_INC  = $$VCPKG_INC)
message([DBG] VCPKG_LIBR = $$VCPKG_LIBR)
message([DBG] VCPKG_LIBD = $$VCPKG_LIBD)

QMAKE_LFLAGS_DEBUG   += /VERBOSE:LIB
QMAKE_LFLAGS_RELEASE += /VERBOSE:LIB

# -------- RELEASE --------
win32:CONFIG(release, debug|release) {
    message([DBG] Build = RELEASE)

    LIBS += -L$$VCPKG_LIBR
    LIBS += -lhdf5 -lhdf5_hl -lhdf5_cpp -lhdf5_hl_cpp
    LIBS += -lzlib -laec
    LIBS += -ldcmdata -lofstd -loflog

    DEST_DLL_DIR = $$OUT_PWD/release
    QMAKE_POST_LINK += cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR" && \
        echo [DBG] Copy DLLs Release to "$$DEST_DLL_DIR" && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\bin\hdf5*.dll"  "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\bin\zlib*.dll"  "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\bin\*aec*.dll"  "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\bin\dcm*.dll"   "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\bin\of*.dll"    "$$DEST_DLL_DIR/" >NUL && \
        echo [DBG] Listing DLLs: && dir /b "$$DEST_DLL_DIR/*.dll"
}

# -------- DEBUG --------
win32:CONFIG(debug, debug|release) {
    message([DBG] Build = DEBUG)

    LIBS += -L$$VCPKG_LIBD
    LIBS += -lhdf5_D -lhdf5_hl_D -lhdf5_cpp_D -lhdf5_hl_cpp_D
    LIBS += -lzlibd -laec
    LIBS += -ldcmdata -lofstd -loflog

    DEST_DLL_DIR = $$OUT_PWD/debug
    QMAKE_POST_LINK += cmd /c if not exist "$$DEST_DLL_DIR" mkdir "$$DEST_DLL_DIR" && \
        echo [DBG] Copy DLLs Debug to "$$DEST_DLL_DIR" && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\debug\bin\hdf5*.dll"  "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\debug\bin\zlib*.dll"  "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\debug\bin\*aec*.dll"  "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\debug\bin\dcm*.dll"   "$$DEST_DLL_DIR/" >NUL && \
        xcopy /Y /I /Q "C:\src\vcpkg\installed\x64-windows\debug\bin\of*.dll"    "$$DEST_DLL_DIR/" >NUL && \
        echo [DBG] Listing DLLs: && dir /b "$$DEST_DLL_DIR/*.dll"
}

