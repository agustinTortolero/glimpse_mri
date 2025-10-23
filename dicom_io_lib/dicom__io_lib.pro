QT       -= gui widgets
TEMPLATE = lib
CONFIG  += c++17 dll
TARGET   = dicom__io_lib

# predictable dirs
MOC_DIR     = $$OUT_PWD/moc
OBJECTS_DIR = $$OUT_PWD/obj
DESTDIR     = $$PWD/bin

HEADERS += \
    include/dicom_api.h

SOURCES += \
    src/dicom_io_lib.cpp

DEFINES += DICOM_IO_LIB_BUILD
DEFINES += _CRT_SECURE_NO_WARNINGS

# ===== vcpkg (explicit triplet paths + debug prints) =====
VCPKG_INSTALLED = C:/src/vcpkg/installed
VCPKG_TRIPLET   = x64-windows

DCMTK_INC = $$VCPKG_INSTALLED/$$VCPKG_TRIPLET/include
DCMTK_LIB = $$VCPKG_INSTALLED/$$VCPKG_TRIPLET/debug/lib

INCLUDEPATH += $$DCMTK_INC
LIBS       += -L$$DCMTK_LIB

message("[DBG][DLL] DCMTK_INC = $$DCMTK_INC")
message("[DBG][DLL] DCMTK_LIB = $$DCMTK_LIB")

# DCMTK (Debug)
LIBS += -ldcmimgle -ldcmimage -ldcmdata -loflog -lofstd
# If you need compressed DICOMs, also add:
# LIBS += -ldcmjpeg -lcharls -ljpeg -lpng -ltiff -lz
