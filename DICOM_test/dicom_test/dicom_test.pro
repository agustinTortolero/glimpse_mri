QT += core
CONFIG += console c++17
CONFIG -= app_bundle
TEMPLATE = app

VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
VCPKG_INC  = $$VCPKG_ROOT/include
VCPKG_LIBR = $$VCPKG_ROOT/lib
VCPKG_LIBD = $$VCPKG_ROOT/debug/lib
VCPKG_BINR = $$VCPKG_ROOT/bin
VCPKG_BIND = $$VCPKG_ROOT/debug/bin

INCLUDEPATH += $$VCPKG_INC

win32:CONFIG(release, debug|release) {
    message([DBG] DCMTK Build=RELEASE)
    LIBS += -L$$VCPKG_LIBR
    LIBS += -ldcmdata -lofstd -loflog
    LIBS += -lzlib

    DEST_DLL_DIR = $$OUT_PWD/release
    QMAKE_POST_LINK += cmd /c echo [DBG] Copy DCMTK DLLs Release... && \
        xcopy /Y /I /Q \"$$VCPKG_BINR\\dcm*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BINR\\of*.dll\"   \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BINR\\zlib*.dll\" \"$$DEST_DLL_DIR\\\"  >NUL
}

win32:CONFIG(debug, debug|release) {
    message([DBG] DCMTK Build=DEBUG)
    LIBS += -L$$VCPKG_LIBD
    LIBS += -ldcmdata -lofstd -loflog       # dcmtk debug libs keep same names
    LIBS += -lzlibd

    DEST_DLL_DIR = $$OUT_PWD/debug
    QMAKE_POST_LINK += cmd /c echo [DBG] Copy DCMTK DLLs Debug... && \
        xcopy /Y /I /Q \"$$VCPKG_BIND\\dcm*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BIND\\of*.dll\"   \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BIND\\zlib*.dll\" \"$$DEST_DLL_DIR\\\"  >NUL
}

QMAKE_LFLAGS_DEBUG   += /VERBOSE:LIB
QMAKE_LFLAGS_RELEASE += /VERBOSE:LIB
DEFINES += _CRT_SECURE_NO_WARNINGS

SOURCES += main.cpp
